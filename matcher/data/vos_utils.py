import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

def dummy_collate_fn(batch):
    return batch


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)

def pad_to_square_tensor(img_tensor, target_size=1024):
    """
    Resize and pad an image tensor [C, H, W] to a square (target_size x target_size),
    adding padding only to the **bottom and right**.

    Args:
        img_tensor (torch.Tensor): Input image tensor of shape [C, H, W].
        target_size (int): Desired output size (square).

    Returns:
        img_padded (torch.Tensor): Padded image tensor of shape [C, target_size, target_size].
        offset_xy (tuple): (left_pad = 0, top_pad = 0) always.
        scale (float): Resize scale applied to original image.
    """
    C, H, W = img_tensor.shape

    # Resize preserving aspect ratio
    scale = min(target_size / H, target_size / W)
    new_H, new_W = int(H * scale), int(W * scale)
    img_resized = F.interpolate(img_tensor.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False)[0]

    # Compute padding amounts (only bottom and right)
    pad_bottom = target_size - new_H
    pad_right = target_size - new_W

    padding = (0, pad_right, 0, pad_bottom)  # pad_left, pad_right, pad_top, pad_bottom
    img_padded = F.pad(img_resized, padding, mode='constant', value=0)

    return img_padded, (pad_bottom, pad_right), scale


def all_to_onehot(masks, labels):
    if len(masks.shape) == 3:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
    else:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1]), dtype=np.uint8)

    for ni, l in enumerate(labels):
        Ms[ni] = (masks == l).astype(np.uint8)
        
    return Ms


class MaskMapper:
    """
    This class is used to convert a indexed-mask to a one-hot representation.
    It also takes care of remapping non-continuous indices
    It has two modes:
        1. Default. Only masks with new indices are supposed to go into the remapper.
        This is also the case for YouTubeVOS.
        i.e., regions with index 0 are not "background", but "don't care".

        2. Exhaustive. Regions with index 0 are considered "background".
        Every single pixel is considered to be "labeled".
    """
    def __init__(self):
        self.labels = []
        self.remappings = {}

        # if coherent, no mapping is required
        self.coherent = True

    def convert_mask(self, mask, exhaustive=False):
        # mask is in index representation, H*W numpy array
        labels = np.unique(mask).astype(np.uint8)
        labels = labels[labels!=0].tolist()

        new_labels = list(set(labels) - set(self.labels))
        if not exhaustive:
            assert len(new_labels) == len(labels), 'Old labels found in non-exhaustive mode'

        # add new remappings
        for i, l in enumerate(new_labels):
            self.remappings[l] = i+len(self.labels)+1
            if self.coherent and i+len(self.labels)+1 != l:
                self.coherent = False

        if exhaustive:
            new_mapped_labels = range(1, len(self.labels)+len(new_labels)+1)
        else:
            if self.coherent:
                new_mapped_labels = new_labels
            else:
                new_mapped_labels = range(len(self.labels)+1, len(self.labels)+len(new_labels)+1)

        self.labels.extend(new_labels)
        mask = torch.from_numpy(all_to_onehot(mask, self.labels)).float()

        # mask num_objects*H*W
        return mask, new_mapped_labels


    def remap_index_mask(self, mask):
        # mask is in index representation, H*W numpy array
        if self.coherent:
            return mask

        new_mask = np.zeros_like(mask)
        for l, i in self.remappings.items():
            new_mask[mask==i] = l
        return new_mask


import math
import heapq
from copy import deepcopy

class Frame():
    def __init__(
        self,
        obj,
        frame_id,
        score
    ):
        self.score_decayed = score
        self.obj = obj
        self.frame_id = frame_id
        self.score = score
        
    def __lt__(self, other):
        a = (
            self.score_decayed,
            self.score,
            self.frame_id
        )
        b = (
            other.score_decayed,
            other.score,
            other.frame_id
        )
        return a < b
        
class Memory():
    def __init__(
        self,
        memory_len = 1,
        fix_first_frame = False,
        fix_last_frame = False,
        memory_decay_ratio = 20,
        memory_decay_type = 'cos'
    ):
        self.memory_len = memory_len - fix_last_frame - fix_first_frame
        assert self.memory_len >= 0
        self.memory = []
        self.fix_first_frame = fix_first_frame
        self.first_frame = None
        self.fix_last_frame = fix_last_frame
        self.last_frame = None
        self.score_decay = memory_decay_ratio

        if memory_decay_ratio:
            if memory_decay_type == 'cos':
                self.score_decay_table = [max(0, math.cos(x/memory_decay_ratio)) for x in range(15)]
            elif memory_decay_type == 'linear':
                self.score_decay_table = [max(0, 1-x/memory_decay_ratio) for x in range(int(memory_decay_ratio))]
            elif memory_decay_type == 'ellipse':
                self.score_decay_table = [max(0, (1-(x/memory_decay_ratio)**2)**0.5) for x in range(int(memory_decay_ratio))]
            elif memory_decay_type == 'exp':
                self.score_decay_table = [max(0, math.exp(-x/memory_decay_ratio)) for x in range(15)]
            elif memory_decay_type == 'constant':
                self.score_decay_table = [1] * 15
                

    def _get_score_decay_ratio(self, x):
        if not self.score_decay:
            return 1
        elif x < len(self.score_decay_table):
            return self.score_decay_table[x]
        else:
            return 0
        
    def update_memory(self, frame):
        if self.fix_first_frame and self.first_frame is None:
            self.first_frame = frame
            return
        
        if self.fix_last_frame:
            if self.last_frame is not None:
                heapq.heappush(self.memory, self.last_frame)
            self.last_frame = frame
        else:
            heapq.heappush(self.memory, frame)
            
        for i in range(len(self.memory)):
            score_decay_ratio = self._get_score_decay_ratio(frame.frame_id - self.memory[i].frame_id)
            score_decayed = score_decay_ratio * self.memory[i].score
            self.memory[i].score_decayed = score_decayed
        heapq.heapify(self.memory)
        if len(self.memory) > self.memory_len:
            heapq.heappop(self.memory)
    
    def get_memory(self):
        memory = deepcopy(self.memory)
        if self.first_frame is not None:
            memory.append(self.first_frame)
            
        if self.last_frame is not None:
            memory.append(self.last_frame)
        
        return memory
        
    def clear_memory(self):
        self.last_frame = None
        self.memory = []