import glob
import torch
import tifffile
import numpy as np

class CustomDataGenerator(torch.utils.data.Dataset):

    def __init__(self, root_dir, subpixels=9, transform=None, train=True):
        self.train = train                                                      # returns image file name when True
        self.dir = root_dir
        self.subpixels = subpixels
        self.transform = transform                                              # image transforms for data augmentation
        self.images = sorted(glob.glob(self.dir + '/data_A/comet_*.tif'))

    def __len__(self):
        return len(self.images) * self.subpixels

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # send noise instead
        if np.random.random() < 0.1:
            return self.noisy_image(idx)
        else:
            return self.comet_image(idx)

    def comet_image(self, idx):
        # make subpixel as one-hot label
        subpixel = idx // len(self.images)
        subp_label = np.zeros((self.subpixels, 1))
        subp_label[subpixel] = 1
        
        # read image
        fname = self.images[idx % len(self.images)]
        fname = fname.replace('data_A', f'data_{chr(65+subpixel)}')
        img = tifffile.imread(fname)                                            # meta (previously facebook :P)
        
        # assuming the comet/polymer tip is centered in the image
        size = len(img)
        tip = (size//2, size//2)
        
        sample = {'image': img, 'tip': tip, 'subpixel': subp_label}

        if self.transform:
            sample = self.transform(sample)
            img, tip, subp_label = sample['image'], sample['tip'], sample['subpixel']

        # normalize image
        img = img.astype(np.float32)
        # img = (img - img.min()) / (img.max() - img.min())
        img = img / 1000
        img = np.expand_dims(img, axis=0)                                       # add channel dimension
        
        # treat tip as a segmentation mask
        mask = np.zeros_like(img[0])
        mask[tip[0], tip[1]] = 1

        subp_label = subp_label.reshape(-1)
        # add last label for no-comet
        subp_label = np.hstack([subp_label, [0]])

        # subp_label = np.array([1, 0])
        if self.train:
            return img, mask, subp_label
        return img, np.array(tip), np.argmax(subp_label), fname.split('/')[-1][:-4]

    def noisy_image(self, idx):
        # make subpixel as one-hot label
        subpixel = idx // len(self.images)
        subp_label = np.zeros((self.subpixels, 1))
        subp_label[subpixel] = 1
        
        # read image
        fname = self.images[idx % len(self.images)]
        fname = fname.replace('data_A', f'data_{chr(65+subpixel)}')
        fname = fname.replace('comet', 'noise')
        img = tifffile.imread(fname)                                            # meta (previously facebook :P)
        
        tip = (0, 0)
        sample = {'image': img, 'tip': tip, 'subpixel': subp_label}

        if self.transform:
            sample = self.transform(sample)
            img = sample['image']

        # normalize image
        img = img.astype(np.float32)
        # img = (img - img.min()) / (img.max() - img.min())
        img = img / 1000
        img = np.expand_dims(img, axis=0)                                       # add channel dimension
        
        # treat tip as a segmentation mask
        mask = np.zeros_like(img[0])
        
        subp_label = np.zeros((self.subpixels+1, 1))
        subp_label[-1] = 1
        subp_label = subp_label.reshape(-1)

        # subp_label = np.array([0, 1])
        if self.train:
            return img, mask, subp_label
        return img, np.array(tip), np.argmax(subp_label), fname.split('/')[-1][:-4]

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
        # img = (img - img.min()) / (img.max() - img.min())
        img = img / 1000
        img = np.expand_dims(img, axis=0)                                       # add channel dimension
        
        return img, fname.split('/')[-1][:-4]