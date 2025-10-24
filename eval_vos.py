import os
import argparse
import shutil
import queue
import math
import numpy as np
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
import datetime
from os.path import join
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
import random
from detectron2.structures import BitMasks, Instances
from torchvision import transforms

from matcher.data.vos_data import DAVISTestDataset, YouTubeVOSTestDataset
from matcher.data.vos_utils import MaskMapper, Memory, Frame, pad_to_square_tensor, MaybeToTensor, dummy_collate_fn


def wrap_instances(mask, id, resize_size):
    # ref
    ref_masks = pad_to_square_tensor(mask, resize_size)
    ref_instances = Instances(resize_size)
    ref_instances.gt_classes = torch.tensor([id], dtype=torch.int64)
    ref_instances.gt_masks = ref_masks
    ref_instances.gt_boxes = BitMasks(ref_masks).get_bounding_boxes()
    ref_instances.ins_ids = torch.tensor([id], dtype=torch.int64)
    
    return ref_instances
    

def wrap_data_ref(batch, args):
    # transforms for image encoder
    encoder_transform = transforms.Compose([
        MaybeToTensor(),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    ref_dict = {}

    # ref
    ref_img, _, _ = pad_to_square_tensor(encoder_transform(batch['support_imgs'])[0, 0], args.image_size)
    ref_image_shape = ref_img.shape[-2:]

    ref_dict["image"] = ref_img

    # label
    ref_dict['height'], ref_dict['width'] = ref_image_shape
    ref_mask_num = batch['support_masks'].shape[1]
    ref_instances = Instances(ref_img.shape[-2:])
    ref_instances.gt_classes = batch['class_id']
    ref_masks = pad_to_square_tensor(batch['support_masks'][0], args.image_size)[0]
    ref_masks = BitMasks(ref_masks)
    ref_instances.gt_masks = ref_masks.tensor
    ref_instances.gt_boxes = ref_masks.get_bounding_boxes()
    ref_instances.ins_ids = batch['class_id'] # TODO
    ref_dict["instances"] = ref_instances

    return ref_dict


def wrap_data_tar(batch, args):
    # transforms for image encoder
    encoder_transform = transforms.Compose([
        MaybeToTensor(),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    tar_dict = {}

    tar_img, (pad_bottom, pad_right), _ = pad_to_square_tensor(encoder_transform(batch['query_img'][0]), args.image_size)
    tar_dict["image"] = tar_img
    # # label
    tar_dict['height'], tar_dict['width'] = tar_img.shape[-2:]
    tar_dict['resize_height'], tar_dict['resize_width'] = tar_img.shape[-2:]
    tar_dict['pad'] = (pad_bottom, pad_right)
    return tar_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ReferFormer training and evaluation script')
    parser.add_argument('--dinov2-size', type=str, default="vit_large")
    parser.add_argument('--sam-size', type=str, default="vit_h")
    parser.add_argument('--dinov2-weights', type=str, default="models/dinov2_vitl14_pretrain.pth")
    parser.add_argument('--sam-weights', type=str, default="models/sam_vit_h_4b8939.pth")
    parser.add_argument('--benchmark', type=str, default='davis17')
    parser.add_argument('--name_exp', type=str, default='eval_davis17')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--image_size', default=1024, type=int)
    parser.add_argument('--img-size', type=int, default=1024)

    parser.add_argument('--num_frame', default=6, type=int)
    parser.add_argument('--max_vid', default=100, type=int)
    args = parser.parse_args()
    args.fix_first_frame = True
    args.output_dir = join(args.output_dir, args.name_exp)
    os.makedirs(args.output_dir, exist_ok=True)
    is_youtube, is_davis = args.benchmark.startswith('ytvos'), args.benchmark.startswith('davis')

    if is_youtube:
        meta_dataset = YouTubeVOSTestDataset(data_root='datasets/YouTube2018', split='valid')
    elif is_davis:
        if args.benchmark == 'davis16':
            # Set up Dataset, a small hack to use the image set in the 2017 folder because the 2016 one is of a different format
            imset_root=os.path.split(os.path.abspath(__file__))[0].replace('/tools', '')
            meta_dataset = DAVISTestDataset('datasets/DAVIS/2016', imset=os.path.join(imset_root, 'datasets/DAVIS/2017/trainval/ImageSets/2016/val.txt'))
        elif args.benchmark == 'davis17':
            video_num = 30
            fold_split = [0, video_num]
            video_indices = (fold_split[0], fold_split[1])
            meta_dataset = DAVISTestDataset('datasets/DAVIS/2017/trainval', imset='2017/val.txt', indices=video_indices)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    meta_loader = meta_dataset.get_datasets()
    sampler_val = SequentialSampler(meta_dataset)
    data_loader_val = DataLoader(
        meta_dataset, batch_size=1, sampler=sampler_val, drop_last=False, num_workers=0,
        collate_fn=dummy_collate_fn
    )
    args.device = 'cuda'
    from matcher.GFSAM import build_model
    model = build_model(args)
    device = torch.device('cuda')

    total_process_time = 0
    total_frames = 0

    J_score = []
    eval_video_names = None

    # Start eval
    for i, vid_reader in tqdm(zip(range(args.max_vid), data_loader_val), total=min(args.max_vid, len(data_loader_val))):
        vid_reader = vid_reader[0]
        loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=0)
        vid_name = vid_reader.vid_name
        vid_length = len(loader)
        
        if eval_video_names and vid_name not in eval_video_names:
            continue
        this_out_path = join(args.output_dir, vid_name)
        if os.path.isdir(this_out_path):
            if len(os.listdir(this_out_path)) == len(loader):
                print(f'[{args.benchmark} ]Video {vid_name} already processed, skipping.... [!!!]')
                continue
        mapper = MaskMapper()
        prompt, prompt_target = None, None
        first_mask_loaded = False
        memory = {}
        n_obj = 0
        for frame_idx, data in enumerate(tqdm(loader, ncols=80)):
            rgb = data['rgb'].cuda()[0]
            msk = data.get('mask')
            info = data['info']
            frame = info['frame'][0]
            shape = info['shape']
            need_resize = info['need_resize'][0]
            
            frame_id = int(frame.strip(".jpg"))

            # Map possibly non-continuous labels to continuous ones
            if frame_idx == 0:
                assert msk is not None
                msk, labels = mapper.convert_mask(msk[0].numpy(), exhaustive=False)
                msk = torch.Tensor(msk).cuda()
                msk_idx = (torch.tensor(labels, device=msk.device) - 1).to(torch.int64)
                msk = msk[msk_idx]
                if need_resize:
                    msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
                support_img = rgb[None, None, ...]
                support_mask = msk[None]
                batch = dict(
                    support_imgs=support_img,
                    support_masks=support_mask,
                    class_id=labels
                )
                ref_dict = wrap_data_ref(batch, args)
                n_obj += len(labels)

            query_img = rgb[None, ...]
            query_mask = data['mask']
            batch = dict(
                query_img=query_img,
                query_mask=query_mask,
            )
            tar_dict = wrap_data_tar(batch, args)
            imgs = torch.cat([ref_dict['image'][None], tar_dict['image'][None]])[None] # b t c h w
            inst_masks = []
            n_insts = len(ref_dict['instances'])
            fs = 1024
            for num_i in range(n_insts):
                if len(data['mask'][data['mask'] == num_i+1]) == 0:
                    output = torch.zeros(1, fs, fs).bool().to(device)
                else:
                    # sup_masks 1, 1, 1024, 1024
                    sup_masks = ref_dict['instances'].gt_masks[num_i:num_i+1,None].float().to(imgs.device)
                    model.set_reference(ref_dict['image'][None, None], sup_masks)
                    model.set_target(tar_dict['image'][None])
                    # 2. Predict mask of target
                    try:
                        output, _ = model.predict()
                    except:
                        output = torch.zeros(1, fs, fs).bool().to(device)
                    model.clear()

                inst_masks.append(output)
                fsize = output.shape[-1]
            pred_masks = torch.stack(inst_masks)[:, 0]
            # pred_masks = (F.interpolate(pred_masks[None].float(), size=imgs.shape[-2:], mode='bilinear', align_corners=False)[0] > 0.5).float()
            pad_y, pad_x = tar_dict['pad']
            if pad_y > 0:
                pred_masks = pred_masks[:, :-pad_y, :]
            if pad_x > 0:
                pred_masks = pred_masks[:, :, :-pad_x]
            # pred_ids = pred['id_seg'].pred_ids
            # pred_masks = torch.zeros_like(pred['id_seg'].pred_masks)
            # pred_scores = [0] * pred['id_seg'].scores.shape[0]
            # resize_size = (tar_dict['resize_height'], tar_dict['resize_width'])

            # # Upsample to original size if needed
            if need_resize:
                pred_masks = (F.interpolate(pred_masks[None].float(), shape, mode='bilinear', align_corners=False)[0] > 0.5).float()

            # Probability mask -> index mask
            pred_masks_sem = pred_masks * torch.tensor(list(range(1, pred_masks.shape[-3] + 1)), device=pred_masks.device)[..., None, None]
            pred_masks_sem = pred_masks_sem.max(dim=0)[0]
            out_mask = pred_masks_sem.cpu().numpy().astype(np.uint8)

            # Save the mask
            if info['save'][0]:
                os.makedirs(this_out_path, exist_ok=True)
                out_mask = mapper.remap_index_mask(out_mask)
                out_img = Image.fromarray(out_mask)
                if vid_reader.get_palette() is not None:
                    out_img.putpalette(vid_reader.get_palette())
                out_img.save(join(this_out_path, frame[:-4]+'.png'))
