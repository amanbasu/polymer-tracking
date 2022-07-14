import torch
import numpy as np
from UNet import UNet
import tifffile
from tqdm import tqdm

IMG_SIZE = 31
THRES = 0.5
LOAD_PATH = '../res/train_output/model_checkpoint_unet.pt'
SAVE_PATH = '/N/slate/amanagar/polymer-tracking/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using:', device)
    
def preprocessing(img):
    img = img.astype(np.float32)
    img = np.clip(img, 0, 1000) / 1000
    return img

def predict(model, test_image):
    global device
    
    t, h, w = test_image.shape
    image = torch.tensor(preprocessing(test_image)).to(device) 
    whole_mask = torch.zeros_like(image).to(device)
    with torch.no_grad():
        for f in tqdm(range(t)):
            stack = []
            for r in range(0, h-IMG_SIZE, IMG_SIZE//3):
                for c in range(0, w-IMG_SIZE, IMG_SIZE//3):
                    clip = image[f, r:r+IMG_SIZE, c:c+IMG_SIZE]
                    clip = torch.unsqueeze(clip, 0)
                    stack += [clip]
            mask, _ = model(torch.stack(stack))
            mask = torch.sigmoid(mask)
            idx = 0
            for r in range(0, h-IMG_SIZE, IMG_SIZE//3):
                for c in range(0, w-IMG_SIZE, IMG_SIZE//3):
                    clip = mask[idx][0]
                    whole_mask[f, r:r+IMG_SIZE, c:c+IMG_SIZE] += clip
                    idx += 1
    whole_mask = whole_mask.cpu().numpy()
    return whole_mask

if __name__ == '__main__':

    # plug-in your model here
    model = UNet(channels=1, classes=1, subpixels=9).to(device)  
    checkpoint = torch.load(LOAD_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_image = tifffile.imread(SAVE_PATH + 'Eb1WT100mscrop1.tif')
    mask = predict(model, test_image)
    tifffile.imwrite(
        SAVE_PATH + 'Eb1WT100mscrop1_mask.tif',
        data=mask
    )