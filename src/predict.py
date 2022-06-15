import sys
import torch
import numpy as np
from tqdm import tqdm
from transform import *
from torchvision import transforms
from torch.utils.data import DataLoader
from generator import CustomDataGenerator
from UNet import UNet
import tifffile

BATCH_SIZE = 1024
LOAD_PATH = '../res/train_output/model_checkpoint_unet.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using:', device)

def get_dataloader():
    # replace with test
    valDataset = CustomDataGenerator(
        '../images/val',
        transform=transforms.Compose([
            RandomCrop(0.5),
            RandomFlip(0.5),
        ]),
        fname=True
    )
    valLoader = DataLoader(
        valDataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=4
    )
    return iter(valLoader)

def save_predictions(image, mask, mask_gt, label, label_gt, fname):
    base_path = '../res/prediction/'
    with open('../res/predictions.txt', 'w+') as f:
        for i in range(len(fname)):
            tip = np.where(mask[i][0]==mask[i][0].max())
            tip_gt = np.where(mask_gt[i]==mask_gt[i].max())

            subpixel = np.argmax(label[i])
            subpixel_gt = np.argmax(label_gt[i])

            content = '\t'.join(
                [fname[i], f'{tip[0][0]},{tip[1][0]}', 
                f'{tip_gt[0][0]},{tip_gt[1][0]}', 
                chr(subpixel+65), chr(subpixel_gt+65)]
            )
            f.write(content+'\n')

            tifffile.imwrite(
                base_path + f'{fname[i]}_{chr(subpixel_gt+65)}.tif',
                data=image[i][0]
            )
    
def predict(model):
    global BATCH_SIZE, device
    
    valLoader = get_dataloader()
    model.eval()
    with torch.no_grad():
        try:
            for image, mask_gt, label_gt, fname in tqdm(valLoader):
                image = image.to(device)                                                  

                mask, label = model(image)
                save_predictions(
                    image.cpu().numpy(), 
                    mask.cpu().numpy(), 
                    mask_gt.numpy(), 
                    label.cpu().numpy(), 
                    label_gt.numpy(),
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