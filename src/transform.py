import cv2
import numpy as np

class RandomFlip(object):
    '''
    Flips the image horizontally (mode=0), vertically (mode=1), 
    or diagonally (mode=2).
    '''

    def __init__(self, prob=0.5):
        assert isinstance(prob, (float, tuple))
        self.prob = prob

    def __call__(self, sample):
        image, tip, subp = sample['image'], sample['tip'], sample['subpixel']

        if np.random.random() < self.prob:
            mode = np.random.choice([0, 1, 2])
            image, tip, subp = self.flip(image, tip, subp, mode=mode)
            
        sample['image'], sample['tip'], sample['subpixel'] = image, tip, subp
        return sample

    def flip(self, img, tip, subpixel, mode=0):
        # mode 0: horizontal flip, 1: vertical flip, 2: diagonal flip
        h, w = img.shape
        
        subpixel = np.array(subpixel).reshape([int(len(subpixel)**0.5)] * 2)
        point = [np.where(subpixel==1)[0][0], np.where(subpixel==1)[1][0]]

        if mode == 0:
            flip = cv2.flip(img, 1)
            tip = (tip[0], w-tip[1]-1)
            point[1] = len(subpixel) - point[1] - 1
        elif mode == 1:
            flip = cv2.flip(img, 0)
            tip = (h-tip[0]-1, tip[1])
            point[0] = len(subpixel) - point[0] - 1
        else:
            flip = cv2.flip(cv2.flip(img, 1), 0)
            tip = (h-tip[0]-1, w-tip[1]-1)
            point[0] = len(subpixel) - point[0] - 1
            point[1] = len(subpixel) - point[1] - 1
        
        subpixel[:] = 0
        subpixel[tuple(point)] = 1
        
        return flip, tip, subpixel.ravel()

class RandomCrop(object):
    '''
    Crops the image randomly in a sample.
    '''

    def __init__(self, size, prob=0.5, pad=5):
        assert isinstance(prob, (float, tuple))
        self.size = size
        self.prob = prob
        self.pad = pad

    def __call__(self, sample):
        image, tip = sample['image'], sample['tip']
        h, _ = image.shape

        if np.random.random() < self.prob:
            # random crop
            starti, startj = np.random.randint(
                self.pad, h - self.size - (self.pad*2), size=2
            )         
        else:
            # center crop
            starti, startj = (h - self.size)//2, (h - self.size)//2            
    
        endi, endj = starti + self.size, startj + self.size            
        image = image[starti:endi, startj:endj]
        tip = (tip[0] - starti, tip[1] - startj)

        sample['image'], sample['tip'] = image, tip
        return sample

class Overlap(object):
    '''
    Overlaps two comets
    '''

    def __init__(self, size):
        self.crop = RandomCrop(size, 1.0)
        self.flip = RandomFlip(1.0)

    def __call__(self, sample):

        image1, tip1 = sample['image1'], sample['tip1']
        image2, tip2 = sample['image2'], sample['tip2']
        mode1, mode2 = np.random.choice(3, replace=False, size=2)

        sp = self.crop({'image': image1, 'tip': tip1})
        imgc, p1 = sp['image'], sp['tip']
        subp = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        
        if np.random.random() < 0.75:
            img1, p1, _ = self.flip.flip(imgc, p1, subp, mode=mode1)
        else:
            img1, p1 = imgc, p1

        sp = self.crop({'image': image2, 'tip': tip2})
        imgc, p2 = sp['image'], sp['tip']
        subp = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        img2, p2, _ = self.flip.flip(imgc, p2, subp, mode=mode2)
        
        out = (img1 + img2)*0.7
        return {'image': out.astype('uint16'), 'tip': [p1, p2]} 