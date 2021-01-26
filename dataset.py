import os
import torch
import torch.utils.data as data
import random
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize


class dataset_unpair(data.Dataset):
    def __init__(self, opts):
        # we need raw images, box positions and cloth mask
        # so the dataroot need to be root dir and the structure are like this
        # dataroot
        # |---trainA
        # |------images raw images and npy data
        # |------masks
        # |---trainB
        # |------images raw images and npy data
        # |------masks
        self.dataroot = opts.dataroot

        # A
        dir_A_path = os.path.join(self.dataroot, opts.phase + 'A')
        files_A = os.listdir(os.path.join(dir_A_path, 'images'))
        images_A = [file for file in files_A if file.endswith('jpg')]
        # images_A = os.listdir(os.path.join(self.dataroot, opts.phase + 'A'))
        self.A = sorted([os.path.join(dir_A_path, 'images', x) for x in images_A])
        # self.A_boxes = [os.path.join(dir_A_path, 'images', str(x.split('.')[0]) + '.npy') for x in images_A]
        # self.A_maskes = [os.path.join(dir_A_path, 'masks', str(x.split('.')[0]) + '.png') for x in images_A]
        # self.A = [os.path.join(self.dataroot, opts.phase + 'A', x) for x in images_A]

        # B
        dir_B_path = os.path.join(self.dataroot, opts.phase + 'B')
        files_B = os.listdir(os.path.join(dir_B_path, 'images'))
        images_B = [file for file in files_B if file.endswith('jpg')]
        # images_A = os.listdir(os.path.join(self.dataroot, opts.phase + 'A'))
        self.B = sorted([os.path.join(dir_B_path, 'images', x) for x in images_B])
        # self.B_boxes = [os.path.join(dir_B_path, 'images', str(x.split('.')[0]) + '.npy') for x in images_B]
        # self.B_maskes = [os.path.join(dir_B_path, 'masks', str(x.split('.')[0]) + '.png') for x in images_B]
        # self.A = [os.path.join(self.dataroot, opts.phase + 'A', x) for x in images_A]

        self.A_size = len(self.A)
        self.B_size = len(self.B)
        self.dataset_size = max(self.A_size, self.B_size)
        self.input_dim_A = opts.input_dim_x
        self.input_dim_B = opts.input_dim_y

        # setup image transformation
        transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
        # if opts.phase == 'train':
        #     transforms.append(RandomCrop(opts.crop_size))
        # else:
        #     transforms.append(CenterCrop(opts.crop_size))
        # if not opts.no_flip:
        #     transforms.append(RandomHorizontalFlip())
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)

        transform_list = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
        transform_list += [ToTensor()]
        self.mask_transforms = Compose(transform_list)
        print('A: %d, B: %d images' % (self.A_size, self.B_size))
        return

    def __getitem__(self, index):
        if self.dataset_size == self.A_size:
            index_A = index
            index_B = random.randint(0, self.B_size - 1)
        else:
            index_A = random.randint(0, self.A_size - 1)
            index_B = index

        A_path = self.A[index_A]
        B_path = self.B[index_B]
        A_mask_path = A_path.replace('images', 'masks').replace('.jpg', '.png')
        B_mask_path = B_path.replace('images', 'masks').replace('.jpg', '.png')
        # A_box_path = A_path.replace('.jpg', '.npy')
        # B_box_path = B_path.replace('.jpg', '.npy')

        A = self.load_img(A_path, self.input_dim_A)
        B = self.load_img(B_path, self.input_dim_B)
        if os.path.isfile(A_mask_path) and os.path.isfile(B_mask_path):
            A_mask = Image.open(A_mask_path).convert('L')
            B_mask = Image.open(B_mask_path).convert('L')

            A_mask = self.mask_transforms(A_mask) * 255.0
            B_mask = self.mask_transforms(B_mask) * 255.0
            # process the mask to get cloth mask, all pixels that satisfy this statement are 1, others are zero
            # 0 'Background', 1 'Hat', 2 'Hair', 3 'Glove', 4 'Sunglasses', 5 'Upper-clothes', 6 'Dress', 7 'Coat',
            # 8 'Socks', 9 'Pants', 10 'Jumpsuits', 11 'Scarf', 12 'Skirt', 13 'Face', 14 'Left-arm', 15 'Right-arm',
            # 16 'Left-leg', 17 'Right-leg', 18 'Left-shoe', 19 'Right-shoe'
            cloth_mask_A_numpy = (A_mask.cpu().numpy() == 5).astype(np.int)
            cloth_mask_B_numpy = (B_mask.cpu().numpy() == 6).astype(np.int)
            # cloth_mask_A_numpy = (A_mask.cpu().numpy() == 12).astype(np.int)
            # cloth_mask_B_numpy = (B_mask.cpu().numpy() == 9).astype(np.int)
            # cloth_mask_A_numpy = (A_mask.cpu().numpy() == 5).astype(np.int) + (A_mask.cpu().numpy() == 6).astype(
            #     np.int) + (A_mask.cpu().numpy() == 7).astype(np.int)
            # cloth_mask_B_numpy = (B_mask.cpu().numpy() == 5).astype(np.int) + (B_mask.cpu().numpy() == 6).astype(
            #     np.int) + (B_mask.cpu().numpy() == 7).astype(np.int)
            A_mask = torch.FloatTensor(cloth_mask_A_numpy)
            B_mask = torch.FloatTensor(cloth_mask_B_numpy)
            # need to normalize
            trans = Compose([Normalize((0.5,), (0.5,))])
            A_mask = trans(A_mask)
            B_mask = trans(B_mask)

        else:
            assert 1 == 2, 'not found mask, plz check your dataset!'
        # TODO: when the resize of img changed, the changes also should happen in box!
        # A_box = np.load(A_box_path)
        # B_box = np.load(B_box_path)

        # A_pos = (A_box[0], A_box[1], A_box[2], A_box[3])
        # A_crop = Image.open(A_path).convert('RGB').crop(A_pos)
        # A_crop = self.transforms(A_crop)
        # B_pos = (B_box[0], B_box[1], B_box[2], B_box[3])
        # B_crop = Image.open(B_path).convert('RGB').crop(B_pos)
        # B_crop = self.transforms(B_crop)

        return {
            'A': A, 'B': B, 'A_mask': A_mask, 'B_mask': B_mask,
        }


    def load_img(self, img_name, input_dim):
        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        if input_dim == 1:
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img

    def __len__(self):
        return self.dataset_size

class dataset_mhp(data.Dataset):
    def __init__(self, opts):
        # we need raw images, box positions and cloth mask
        # so the dataroot need to be root dir and the structure are like this
        # dataroot
        # |---trainA
        # |------images raw images and npy data
        # |------masks
        # |---trainB
        # |------images raw images and npy data
        # |------masks
        self.dataroot = opts.dataroot

        # A
        dir_A_path = os.path.join(self.dataroot, opts.phase + 'A')
        files_A = os.listdir(dir_A_path)
        images_A = [file for file in files_A if file.endswith('png')]
        self.A = sorted([os.path.join(dir_A_path, x) for x in images_A])
        # self.A_boxes = [os.path.join(dir_A_path, 'images', str(x.split('.')[0]) + '.npy') for x in images_A]
        # self.A_maskes = [os.path.join(dir_A_path, 'masks', str(x.split('.')[0]) + '.png') for x in images_A]
        # self.A = [os.path.join(self.dataroot, opts.phase + 'A', x) for x in images_A]

        # B
        dir_B_path = os.path.join(self.dataroot, opts.phase + 'B')
        files_B = os.listdir(os.path.join(dir_B_path, 'images'))
        images_B = [file for file in files_B if file.endswith('jpg')]
        # images_A = os.listdir(os.path.join(self.dataroot, opts.phase + 'A'))
        self.B = sorted([os.path.join(dir_B_path, 'images', x) for x in images_B])
        # self.B_boxes = [os.path.join(dir_B_path, 'images', str(x.split('.')[0]) + '.npy') for x in images_B]
        # self.B_maskes = [os.path.join(dir_B_path, 'masks', str(x.split('.')[0]) + '.png') for x in images_B]
        # self.A = [os.path.join(self.dataroot, opts.phase + 'A', x) for x in images_A]

        self.A_size = len(self.A)
        self.B_size = len(self.B)
        self.dataset_size = max(self.A_size, self.B_size)
        self.input_dim_A = opts.input_dim_x
        self.input_dim_B = opts.input_dim_y

        # setup image transformation
        transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
        # if opts.phase == 'train':
        #     transforms.append(RandomCrop(opts.crop_size))
        # else:
        #     transforms.append(CenterCrop(opts.crop_size))
        # if not opts.no_flip:
        #     transforms.append(RandomHorizontalFlip())
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)

        transform_list = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
        transform_list += [ToTensor()]
        self.mask_transforms = Compose(transform_list)
        print('A: %d, B: %d images' % (self.A_size, self.B_size))
        return

    def __getitem__(self, index):
        if self.dataset_size == self.A_size:
            index_A = index
            index_B = random.randint(0, self.B_size - 1)
        else:
            index_A = random.randint(0, self.A_size - 1)
            index_B = index

        A_path = self.A[index_A]
        B_path = self.B[index_B]
        A_mask_path = A_path.replace('images', 'masks').replace('.jpg', '.png')
        B_mask_path = B_path.replace('images', 'masks').replace('.jpg', '.png')
        A_box_path = A_path.replace('.jpg', '.npy')
        B_box_path = B_path.replace('.jpg', '.npy')

        A = self.load_img(A_path, self.input_dim_A)
        B = self.load_img(B_path, self.input_dim_B)
        if os.path.isfile(A_mask_path) and os.path.isfile(B_mask_path):
            A_mask = Image.open(A_mask_path).convert('L')
            B_mask = Image.open(B_mask_path).convert('L')

            A_mask = self.mask_transforms(A_mask) * 255.0
            B_mask = self.mask_transforms(B_mask) * 255.0
            # process the mask to get cloth mask, all pixels that satisfy this statement are 1, others are zero
            # 0 'Background', 1 'Hat', 2 'Hair', 3 'Glove', 4 'Sunglasses', 5 'Upper-clothes', 6 'Dress', 7 'Coat',
            # 8 'Socks', 9 'Pants', 10 'Jumpsuits', 11 'Scarf', 12 'Skirt', 13 'Face', 14 'Left-arm', 15 'Right-arm',
            # 16 'Left-leg', 17 'Right-leg', 18 'Left-shoe', 19 'Right-shoe'
            cloth_mask_A_numpy = (A_mask.cpu().numpy() == 12).astype(np.int)
            cloth_mask_B_numpy = (B_mask.cpu().numpy() == 9).astype(np.int)
            # cloth_mask_A_numpy = (A_mask.cpu().numpy() == 5).astype(np.int) + (A_mask.cpu().numpy() == 6).astype(
            #     np.int) + (A_mask.cpu().numpy() == 7).astype(np.int)
            # cloth_mask_B_numpy = (B_mask.cpu().numpy() == 5).astype(np.int) + (B_mask.cpu().numpy() == 6).astype(
            #     np.int) + (B_mask.cpu().numpy() == 7).astype(np.int)
            A_mask = torch.FloatTensor(cloth_mask_A_numpy)
            B_mask = torch.FloatTensor(cloth_mask_B_numpy)
            # need to normalize
            trans = Compose([Normalize((0.5,), (0.5,))])
            A_mask = trans(A_mask)
            B_mask = trans(B_mask)

        else:
            assert 1 == 2, 'not found mask, plz check your dataset!'
        # TODO: when the resize of img changed, the changes also should happen in box!
        A_box = np.load(A_box_path)
        B_box = np.load(B_box_path)

        A_pos = (A_box[0], A_box[1], A_box[2], A_box[3])
        A_crop = Image.open(A_path).convert('RGB').crop(A_pos)
        A_crop = self.transforms(A_crop)
        B_pos = (B_box[0], B_box[1], B_box[2], B_box[3])
        B_crop = Image.open(B_path).convert('RGB').crop(B_pos)
        B_crop = self.transforms(B_crop)

        return {
            'A': A, 'B': B, 'A_mask': A_mask, 'B_mask': B_mask,
            'A_box': A_box, 'B_box': B_box, 'A_crop': A_crop, 'B_crop': B_crop
        }


    def load_img(self, img_name, input_dim):
        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        if input_dim == 1:
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img

    def __len__(self):
        return self.dataset_size