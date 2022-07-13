from scipy.io import loadmat
import glob
import tifffile
import json
import os
import argparse

metainfo = ['PolAng', 'TipProb', 'ExpTime', 'TipExp', 'eNoise']

def save_tif(ifilename, ofilename, subpixel=0):
    src = loadmat(ifilename, squeeze_me=True)
    
    metadata = {}
    for key in metainfo:
        metadata[key] = src[key]

    extra_tags = [("MicroManagerMetadata", 's', 0, json.dumps(metadata), True)]
    
    tifffile.imwrite(
        ofilename,
        data=src['M'][:, :, subpixel],
        extratags=extra_tags,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments for creating the tif files.'
    )
    parser.add_argument(
        '--path', default='./', type=str, help='base path for images'
    )
    parser.add_argument(
        '--subpixels', default=9, type=int, help='number of subpixels'
    )
    parser.add_argument(
        '--comet', default=True, type=bool, help='whether comet or noise'
    )
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.path, 'comet_*.mat')) if args.comet \
        else glob.glob(os.path.join(args.path, 'noise_*.mat'))
    base_folder = os.path.abspath(args.path).split('/')[-1]
    for ifile in files:
        for sp in range(args.subpixels):
            ofile = ifile.replace(
                base_folder, f'{base_folder}_{chr(sp+65)}'
            ).replace('.mat', '.tif')

            newpath = '/'.join(ofile.split('/')[:-1])
            if not os.path.exists(newpath):
                os.makedirs(newpath)

            # pad digits to 5 characters
            spl = ofile.split('_')
            ofile = '_'.join(spl[:-1]) + '_' + spl[-1].rjust(9, '0')

            save_tif(ifile, ofile, subpixel=sp)