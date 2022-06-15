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

        # make subpixel as one-hot label
        subpixel = idx // len(self.images)
        subp_label = np.zeros((self.subpixels, 1))
        subp_label[subpixel] = 1
        
        # read image
        fname = self.images[idx % len(self.images)]
        fname.replace('data_A', f'data_{chr(65+subpixel)}')
        img, meta = self.read_tif(fname)                                        # meta (previously facebook :P)
        
        # assuming the comet/polymer tip is centered in the image
        size = len(img)
        tip = (size//2, size//2)
        
        sample = {'image': img, 'tip': tip, 'subpixel': subp_label}

        if self.transform:
            sample = self.transform(sample)
            img, tip = sample['image'], sample['tip']

        # normalize image
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        img = np.expand_dims(img, axis=0)                                       # add channel dimension
        
        # treat tip as a segmentation mask
        mask = np.zeros_like(img[0])
        mask[tip[0], tip[1]] = 1

        subp_label = subp_label.reshape(-1)

        if self.train:
            return img, mask, subp_label
        return img, np.array(tip), np.argmax(subp_label), fname.split('/')[-1][:-4]
    
    def read_tif(self, filename):
        frames = tifffile.TiffFile(filename)
        page = frames.pages[0]
        metadata = page.tags["MicroManagerMetadata"].value
        return tifffile.imread(filename), metadata