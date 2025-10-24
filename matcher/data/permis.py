import os

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from os.path import join

class DatasetPerMis(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize=False, num_val=600):
        self.split = split
        self.benchmark = 'permis'
        self.shot = shot
        self.nclass = 1
        datapath = join(datapath, 'burst/PerMIRS')
        self.transform = transform
        self.episodes = []
        for vid_id in tqdm(os.listdir(datapath)):
            masks_np = np.load(f"{datapath}/{vid_id}/masks.npz.npy", allow_pickle=True)

            for f in range(3):
                gt_mask = torch.tensor(list(masks_np[f].values())[0]).int()
                frame_path = f"{datapath}/{vid_id}/{f}.jpg"
                full_img = Image.open(frame_path).convert("RGB")
                if f == 0:
                    first_episode = {"supp_img": [full_img], "supp_mask": [gt_mask]}
                    second_episode = {"supp_img": [full_img], "supp_mask": [gt_mask]}
                if f == 1:
                    first_episode["query_img"] = full_img
                    first_episode["query_mask"] = gt_mask
                if f == 2:
                    second_episode["query_img"] = full_img
                    second_episode["query_mask"] = gt_mask
            self.episodes.append(first_episode)
            self.episodes.append(second_episode)

            self.class_ids = [0]

    def __len__(self):
        # if it is the target domain, then also test on entire dataset
        return len(self.episodes)

    def __getitem__(self, idx):
        episode = self.episodes[idx]
        support_imgs, support_masks = episode["supp_img"], episode["supp_mask"]
        query_img, query_mask = episode["query_img"], episode["query_mask"]

        query_img = self.transform(query_img)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'support_set': (support_imgs, support_masks),
                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'class_id': torch.tensor(0)
                 }

        return batch


    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask


