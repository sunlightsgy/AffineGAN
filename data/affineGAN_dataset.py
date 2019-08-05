import os.path
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from data.base_dataset import BaseDataset


class AffineGANDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def pre_process_img(self, path, convertRGB, w_offset, h_offset, flip):
        if not os.path.exists(path):
            if convertRGB:
                return np.zeros(
                    (self.opt.fineSize, self.opt.fineSize, 3), dtype=np.uint8
                )
            else:
                return np.zeros((self.opt.fineSize, self.opt.fineSize), dtype=np.uint8)

        image = Image.open(path)
        if convertRGB:
            image = image.convert("RGB")
        else:
            image = image.convert("L")
        image = image.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        image = np.array(image)
        if not convertRGB:
            image = image[..., np.newaxis]
            image = np.tile(image, [1, 1, 3])
        image = transforms.ToTensor()(image)
        image = image[
            :,
            h_offset : h_offset + self.opt.fineSize,
            w_offset : w_offset + self.opt.fineSize,
        ]
        image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)
        if not convertRGB:
            image[image < 0.5] = 0.0
            image[image >= 0.5] = 1.0
        if flip:
            idx = torch.LongTensor([i for i in range(image.size(2) - 1, -1, -1)])
            image = image.index_select(2, idx)

        return image

    def initialize(self, opt):
        self.opt = opt
        self.dir_AB = os.path.join(opt.dataroot, opt.phase, "img")
        self.AB_paths = []

        if not self.opt.no_patch:
            self.dir_AB_patch = os.path.join(opt.dataroot, opt.phase, "patch")
            self.AB_patch = []

        video_names = sorted([f for f in os.listdir(self.dir_AB) if "." not in f])
        self.sample_num = len(video_names)

        for sample_idx in range(self.sample_num):
            sample_name = video_names[sample_idx]
            self.AB_paths.append(os.path.join(self.dir_AB, sample_name))
            if not self.opt.no_patch:
                self.AB_patch.append(os.path.join(self.dir_AB_patch, sample_name))

        assert opt.resize_or_crop == "resize_and_crop"

    def __getitem__(self, index):
        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        AB_path = self.AB_paths[index]
        img_names = sorted([f for f in os.listdir(AB_path)])

        flip = (not self.opt.no_flip) and random.random() < 0.5

        A_name = os.path.join(AB_path, img_names[0])
        A = self.pre_process_img(A_name, True, w_offset, h_offset, flip)
        ret_dict = {"A": A, "A_paths": AB_path}

        if self.opt.isTrain:
            if not self.opt.no_patch:
                # When Testing, the model doesn't need patches
                A_name = os.path.join(self.AB_patch[index], img_names[0])
                A_patch = self.pre_process_img(A_name, False, w_offset, h_offset, flip)
                ret_dict["A_patch"] = A_patch
                B_patch_list = []

            B_list = []
            np.random.seed()
            img_sample = range(1, len(img_names))
            img_sample = np.random.choice(
                img_sample, self.opt.train_imagenum, replace=True
            )
            for img_idx in range(self.opt.train_imagenum):
                sample_image_idx = img_sample[img_idx]

                B_name = os.path.join(AB_path, img_names[sample_image_idx])
                B = self.pre_process_img(B_name, True, w_offset, h_offset, flip)

                B_list.append(B)
                if not self.opt.no_patch:
                    B_name = os.path.join(
                        self.AB_patch[index], img_names[sample_image_idx]
                    )
                    B_patch = self.pre_process_img(
                        B_name, False, w_offset, h_offset, flip
                    )
                    B_patch_list.append(B_patch)

            ret_dict["B_list"] = B_list
            if not self.opt.no_patch:
                ret_dict["B_patch_list"] = B_patch_list

        return ret_dict

    def __len__(self):
        return self.sample_num

    def name(self):
        return "AffineGANDataset"
