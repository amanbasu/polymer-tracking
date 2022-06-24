import sys
import torch
import numpy as np
from tqdm import tqdm
from transform import *
from torchvision import transforms
from torch.utils.data import DataLoader
from generator import TestDataGenerator
from UNet import UNet
import tifffile, json

BATCH_SIZE = 1024
IMG_SIZE = 31
LOAD_PATH = '../res/train_output/model_checkpoint_unet.pt'
SAVE_PATH = '../res/test_pred/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using:', device)

def get_dataloader():
    # replace with test
    testDataset = TestDataGenerator(
        '../images/test'
    )
    testLoader = DataLoader(
        testDataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=4
    )
    return iter(testLoader)

def save_predictions(image, mask, label, fname):
    global SAVE_PATH
    for i in range(len(fname)):
        tip = np.where(mask[i][0]==mask[i][0].max())
        subpixel = np.argmax(label[i])

        metadata = {
            'tip': f'{tip[0][0]},{tip[1][0]}',
            'subpixel': chr(subpixel+65),
        }
        extra_tags = [
            ("MicroManagerMetadata", 's', 0, json.dumps(metadata), True)
        ]
        tifffile.imwrite(
            SAVE_PATH + f'{fname[i]}.tif',
            data=image[i][0],
            extratags=extra_tags,
        )
    
def predict(model):
    global BATCH_SIZE, device
    
    testLoader = get_dataloader()
    model.eval()
    with torch.no_grad():
        try:
            for image, fname in tqdm(testLoader):
                image = image.to(device)                                                  
                mask, label = model(image)
                save_predictions(
                    image.cpu().numpy(), 
                    mask.cpu().numpy(), 
                    label.cpu().numpy(), 
                    fname
                )
        except StopIteration:
            pass

if __name__ == '__main__':

    # plug-in your model here
    model = UNet(channels=1, classes=1, subpixels=9).to(device)  
    checkpoint = torch.load(LOAD_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    predict(model)