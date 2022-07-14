import glob
import torch
import tifffile
import numpy as np
from transform import Overlap

class CustomDataGenerator(torch.utils.data.Dataset):
    def __init__(self, root_dir, subpixels=9, imgs=31, transform=None):
        self.dir = root_dir
        self.imgs = imgs
        self.subpixels = subpixels
        self.transform = transform                                              # image transforms for data augmentation
        self.overlap = Overlap(size=imgs)
        self.images = sorted(glob.glob(self.dir + '/data_A/comet_*.tif'))

    def __len__(self):
        return len(self.images) * self.subpixels

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if np.random.random() < 0.3:
            return self.noisy_image(idx)
        elif np.random.random() < 0.8:
            return self.comet_image(idx)
        else:
            return self.overlap_image(idx) 

    def comet_image(self, idx):
        # make subpixel as one-hot label
        subpixel = idx // len(self.images)
        label = np.zeros((self.subpixels, 1))
        label[subpixel] = 1
        
        # read image
        fname = self.images[idx % len(self.images)]
        fname = fname.replace('data_A', f'data_{chr(65+subpixel)}')
        img = tifffile.imread(fname)
         
        # the comet/polymer tip is centered in the image
        size = len(img)
        tip = (size//2, size//2)
        
        sample = {'image': img, 'tip': tip, 'subpixel': label}

        if self.transform:
            sample = self.transform(sample)
            img, tip = sample['image'], sample['tip']
            label = sample['subpixel']

        # normalize image
        img = img.astype(np.float32)
        img = np.clip(img, 0, 1000) / 1000
        img = np.expand_dims(img, axis=0)                                       # add channel dimension
        
        # segmentation mask for comet tip
        mask = np.zeros_like(img[0])
        if tip[0] == 0 or \
            tip[1] == 0 or \
            tip[0] == self.imgs-1 or \
            tip[1] == self.imgs-1:
            # image with no comet tip
            tip = (0, 0)
        else:
            mask[tip[0], tip[1]] = 1

        # add last label for no-comet
        if tip == (0, 0):
            label = np.zeros((self.subpixels+1,))
        else:
            label = label.reshape(-1)
            label = np.hstack([label, [0]])

        return img, mask, label

    def noisy_image(self, idx):
        # make subpixel as one-hot label
        subpixel = idx // len(self.images)
        label = np.zeros((self.subpixels, 1))
        label[subpixel] = 1
        
        # read image
        fname = self.images[idx % len(self.images)]
        fname = fname.replace('data_A', f'data_{chr(65+subpixel)}')
        fname = fname.replace('comet', 'noise')
        img = tifffile.imread(fname)
        
        tip = (0, 0)
        sample = {'image': img, 'tip': tip, 'subpixel': label}

        if self.transform:
            sample = self.transform(sample)
            img = sample['image']

        # normalize image
        img = img.astype(np.float32)
        img = np.clip(img, 0, 1000) / 1000
        img = np.expand_dims(img, axis=0)                                       # add channel dimension
        
        # treat tip as a segmentation mask
        mask = np.zeros_like(img[0])
        
        label = np.zeros((self.subpixels+1, 1))
        label[-1] = 1
        label = label.reshape(-1)

        return img, mask, label

    def overlap_image(self, idx):
        # make subpixel as one-hot label
        subpixel = idx // len(self.images)
        label = np.zeros((self.subpixels, 1))
        label[subpixel] = 1
        
        # read image
        fname = self.images[idx % len(self.images)]
        fname = fname.replace('data_A', f'data_{chr(65+subpixel)}')
        img = tifffile.imread(fname)

        # read another random image
        fname2 = np.random.choice(self.images)
        fname2 = fname2.replace('data_A', f'data_{chr(65+subpixel)}')
        img2 = tifffile.imread(fname2)  
        
        # assuming the comet/polymer tip is centered in the image
        size = len(img)
        tip = (size//2, size//2)
        
        sample = {'image1': img, 'tip1': tip, 'image2': img2, 'tip2': tip}
        sample = self.overlap(sample)
        img, tip = sample['image'], sample['tip']
            
        # normalize image
        img = img.astype(np.float32)
        img = np.clip(img, 0, 1000) / 1000
        img = np.expand_dims(img, axis=0)                                       # add channel dimension
        
        # treat tip as a segmentation mask
        mask = np.zeros_like(img[0])
        mask[tip[0][0], tip[0][1]] = 1
        mask[tip[1][0], tip[1][1]] = 1

        label = label.reshape(-1)
        label = np.hstack([label, [0]])
        
        return img, mask, label

class TestDataGenerator(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.dir = root_dir
        self.transform = transform 
        self.images = sorted(glob.glob(self.dir + '/comet_*.tif'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        # read image
        fname = self.images[idx]
        img = tifffile.imread(fname)
        
        # normalize image
        img = img.astype(np.float32)
        img = np.clip(img, 0, 1000) / 1000
        img = np.expand_dims(img, axis=0)                                       # add channel dimension
        
        return img, fname.split('/')[-1][:-4]