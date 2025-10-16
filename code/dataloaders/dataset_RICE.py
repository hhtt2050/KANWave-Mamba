import torch
import random
import numpy as np
from torchvision import transforms
import itertools
from scipy import ndimage
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

class VOCAgriculture(Dataset):
    CLASSES = ['background', 'disease']
    def __init__(self, root, split='train', transform=None, crop_size=(256, 256)):
        self.root = root
        self.split = split
        self.transform = transform
        self.crop_size = crop_size
        self.image_dir = os.path.join(root, 'JPEGImages')
        self.mask_dir = os.path.join(root, 'SegmentationClass')
        split_file = os.path.join(root, 'ImageSets', 'Segmentation', f'{split}.txt')
        self.images = [line.strip() for line in open(split_file)]

        self.color_jitter = transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, f"{img_name}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{img_name}.png")

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        img = F.to_tensor(img)  # [3, H, W]
        mask = self._preprocess_mask(mask)  # [H, W]


        if self.transform == 'train':
            img, mask = self._train_augmentation(img, mask)
        elif self.transform == 'val':
            img, mask = self._val_preprocessing(img, mask)


        return {'image': img, 'label': mask}

    def _preprocess_mask(self, mask):
        mask_np = np.array(mask)
        mask_bin = mask_np.astype(np.uint8)
        return torch.from_numpy(mask_bin).long()

    def _train_augmentation(self, img, mask):
        """img [3,H,W], mask [H,W]"""
        # 4D [B,C,H,W]
        img = img.unsqueeze(0)  # [1,3,H,W]
        mask = mask.unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]

        h, w = img.shape[-2], img.shape[-1]
        min_scale_h = self.crop_size[0] / h
        min_scale_w = self.crop_size[1] / w
        min_scale = max(min_scale_h, min_scale_w)
        scale = random.uniform(min_scale, 2.0)
        new_h = int(h * scale)
        new_w = int(w * scale)
        img = F.resize(img, (new_h, new_w), interpolation=Image.BILINEAR)
        mask = F.resize(mask, (new_h, new_w), interpolation=Image.NEAREST)

        new_h, new_w = img.shape[-2], img.shape[-1]
        if new_h < self.crop_size[0] or new_w < self.crop_size[1]:
            resize_size = (max(new_h, self.crop_size[0]), max(new_w, self.crop_size[1]))
            img = F.resize(img, resize_size, interpolation=Image.BILINEAR)
            mask = F.resize(mask, resize_size, interpolation=Image.NEAREST)

        i, j, h, w = transforms.RandomCrop.get_params(
            img, output_size=self.crop_size)
        img = F.crop(img, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        if random.random() > 0.5:
            img = self.color_jitter(img.squeeze(0)).unsqueeze(0)  # 保持4D



        angle = random.choice([0, 90, 180, 270])
        img = F.rotate(img, angle)  # [1,3,224,224]
        mask = F.rotate(mask, angle, interpolation=Image.NEAREST)  # [1,1,224,224]


        return img.squeeze(0), mask.squeeze().long()  # [3,224,224], [224,224]

    def _val_preprocessing(self, img, mask):
        """ img [3,H,W], mask [H,W]"""
        # 4D
        img = img.unsqueeze(0)  # [1,3,H,W]
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        img = F.resize(img, self.crop_size, interpolation=Image.BILINEAR).squeeze(0)  # [3,224,224]
        mask = F.resize(mask, self.crop_size, interpolation=Image.NEAREST).squeeze(0).squeeze(0)  # [224,224]

        return img, mask.long()

def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
