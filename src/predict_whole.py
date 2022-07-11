import torch
import numpy as np
from UNet import UNet
import tifffile

IMG_SIZE = 31
LOAD_PATH = '../res/train_output/model_checkpoint_unet.pt'
SAVE_PATH = '../res/test_whole/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using:', device)

def save_predictions(mask, idx):
    global SAVE_PATH

    tifffile.imwrite(
        SAVE_PATH + f'mask_{idx[0]}_{idx[1]}_{idx[2]}.tif',
        data=mask[0]
    )
    
def predict(model, test_image):
    global device
    
    t, h, w = test_image.shape
    model.eval()
    with torch.no_grad():
        for f in range(t):
            for r in range(0, h, IMG_SIZE):
                for c in range(0, w, IMG_SIZE):
                    clip = test_image[f, r:r+IMG_SIZE, c:c+IMG_SIZE]
                    clip = np.expand_dims(clip, axis=(0, 1))
                    clip = clip.to(device) 
                    mask, _ = model(clip)
                    save_predictions(
                        mask.cpu().numpy(),
                        (f, r, c)
                    )

if __name__ == '__main__':

    # plug-in your model here
    model = UNet(channels=1, classes=1, subpixels=8).to(device)  
    checkpoint = torch.load(LOAD_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_image = tifffile.imread('../images/Eb1WT100mscrop1.tif')
    predict(model, test_image)