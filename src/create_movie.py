import os
import argparse
import tifffile
import numpy as np
from PIL import Image
from numba import jit
from tqdm import tqdm

'''
Supresses multiple nearby predictions

Parameters
    mask - numpy.ndarray
        mask containing values between 0 to 255
    thres - float
        threshold for prediction

Returns
    mask_pruned - numpy.ndarray
        pruned mask
'''
@jit(nopython=True)
def nonmax_supression(mask, thres=0.5):
    h, w = mask.shape
    mask_pruned = np.zeros_like(mask)
    tval = mask.max() * thres
    for i in range(1, h-1, 1):
        for j in range(1, w-1, 1):
            if mask[i, j] < tval:
                continue
            crop = mask[i-1:i+2, j-1:j+2]
            tip = np.where(crop==crop.max())                                    # only select the max value in the neighborhood
            mask_pruned[i-1+tip[0][0], j-1+tip[1][0]] = 255
    return mask_pruned

'''
Overlays mask over the image

Parameters
    image - numpy.ndarray
        background image containing values between 0 to 255
    mask - numpy.ndarray
        foreground mask containing values between 0 to 255
    thres - float
        threshold for prediction
    memory - int
        number of previous masks to keep

Returns
    frames - list
        list of all the frames
'''
def make_movie(image, mask, thres=0.5, memory=100):
    frames = []
    nframes, h, w, _ = image.shape
    history = np.zeros((h, w))                                                  # keep a track of all masks over frames
    for f in tqdm(range(nframes)):
        pruned_mask = nonmax_supression(mask[f], thres)
        history[pruned_mask==255] = f

        yam = np.zeros((h, w, 3), dtype='uint8')
        yam[history>max(0, f-memory)] = [0, 0, 255]                             # only keep #memory masks 

        background = Image.fromarray(image[f])
        foreground = Image.fromarray(yam).convert('RGBA')
        foreground.putalpha(100)
        background.paste(foreground, (0, 0), foreground)

        frames += [np.array(background)]
        
    return frames

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments for creating the tracking file from mask.'
    )
    parser.add_argument(
        '--path', default='./', type=str, help='base path for images'
    )
    parser.add_argument(
        '--image', default='Eb1WT100mscrop1.tif', type=str, help='image name'
    )
    parser.add_argument(
        '--mask', default='Eb1WT100mscrop1_mask.tif', type=str, help='mask name'
    )
    parser.add_argument(
        '--thres', default=0.5, type=float, help='prediction threshold'
    )
    parser.add_argument(
        '--out', default='Eb1WT100mscrop1_track.tif', type=str, help='output name'
    )
    
    args = parser.parse_args()
    image = tifffile.imread(os.path.join(args.path, args.image))
    mask = tifffile.imread(os.path.join(args.path, args.mask))

    image_arr = ((image / 1000) * 255).astype('uint8')
    image_arr = np.array([image_arr]*3).transpose((1, 2, 3, 0))

    movie = make_movie(image_arr, mask, args.thres)
    tifffile.imwrite(
        os.path.join(args.path, args.out),
        data=np.array(movie),
    )