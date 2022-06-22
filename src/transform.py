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

    def __init__(self, size, prob=0.5):
        assert isinstance(prob, (float, tuple))
        self.size = size
        self.prob = prob

    def __call__(self, sample):
        image, tip = sample['image'], sample['tip']
        h, _ = image.shape

        if np.random.random() < self.prob:
            # random crop
            starti, startj = np.random.randint(0, h - self.size, size=2)         
        else:
            # center crop
            starti, startj = (h - self.size)//2, (h - self.size)//2            
    
        endi, endj = starti + self.size, startj + self.size            
        image = image[starti:endi, startj:endj]
        tip = (tip[0] - starti, tip[1] - startj)
        
        # if np.random.random() < self.prob:
        #     starti, startj = np.random.randint(0, size//2-4, size=2)
        #     endi, endj = np.random.randint(size//2+3, size, size=2)

        #     imgc = np.clip(
        #         (0.5 + np.random.random(image.shape)) * image.mean(), 
        #         image.min(),
        #         np.median(image)
        #     )
        #     h, w = endi - starti, endj - startj
            
        #     i = np.random.randint(0, size-h)
        #     j = np.random.randint(0, size-w)
            
        #     imgc[i:i+h, j:j+w] = image[starti:endi, startj:endj]
            
        #     image = imgc
        #     tip = (tip[0] - starti + i, tip[1] - startj + j)

        sample['image'], sample['tip'] = image, tip
        return sample