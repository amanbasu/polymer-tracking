import sys
import torch
import numpy as np
from tqdm import tqdm
from transform import *
from torchvision import transforms
from torch.utils.data import DataLoader
from generator import CustomDataGenerator
from UNet import UNet

INIT_EPOCH = 0
EPOCHS = 50
BATCH_SIZE = 1024
IMG_SIZE = 31                                                                  
LEARNING_RATE = 0.0001
STEPS = 9000 // BATCH_SIZE + 1
SAVE_PATH = '../res/train_output/model_checkpoint_unet.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using:', device)

class dice_loss(torch.nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()
        self.smooth = 1.

    def forward(self, logits, labels):
        logf = torch.sigmoid(logits).view(-1)
        labf = labels.view(-1)
        intersection = (logf * labf).sum()

        num = 2. * intersection + self.smooth
        den = logf.sum() + labf.sum() + self.smooth
        return 1 - (num/den)

class combine_loss(torch.nn.Module):
    def __init__(self):
        super(combine_loss, self).__init__()
        self.bceLoss = torch.nn.BCEWithLogitsLoss()
        self.dice_loss = dice_loss()

    def forward(self, logits, labels):
        dice = self.dice_loss(logits[0], labels[0])
        bce = self.bceLoss(logits[1], labels[1].float())
        return dice + bce
            
def get_dataloader():
    trainDataset = CustomDataGenerator(
        '../images/train', 
        transform=transforms.Compose([
            RandomFlip(0.5),
            RandomCrop(IMG_SIZE, 0.8),
        ])
    )
    valDataset = CustomDataGenerator(
        '../images/val',
        transform=transforms.Compose([
            RandomCrop(IMG_SIZE, 0.0),
        ])
    )
    trainLoader = DataLoader(
        trainDataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    valLoader = DataLoader(
        valDataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=4
    )
    return iter(trainLoader), iter(valLoader)

def train(model, criterion, opt, scheduler):
    global INIT_EPOCH, EPOCHS, BATCH_SIZE, SAVE_PATH, device
    
    model.train()
    for epoch in range(INIT_EPOCH, EPOCHS):
        trainLoader, valLoader = get_dataloader()
        # when dataloader runs out of batches, it throws an exception
        try:
            for image, mask_gt, label_gt in tqdm(trainLoader):
                image = image.to(device)                                                  
                mask_gt = mask_gt.to(device)
                label_gt = label_gt.to(device)
                
                opt.zero_grad(set_to_none=True)                                 # clear gradients w.r.t. parameters

                mask, label = model(image)
                loss = criterion([mask, label], [mask_gt, label_gt])

                # regularization
                weight = torch.tensor(0.).to(device) 
                for parameter in model.parameters():
                    weight += torch.norm(parameter)
                l2_loss = 0.001 * weight
                loss += l2_loss

                loss.backward()                                                 # getting gradients
                opt.step()                                                      # updating parameters
                scheduler.step()                                                # to change the learing rate
        except StopIteration:
            pass

        # get model performace on val set
        with torch.no_grad():
            dice, accuracy = [], []
            dloss = dice_loss()
            try:
                for image, mask_gt, label_gt in tqdm(valLoader):
                    image = image.to(device)                                                  
                    mask_gt = mask_gt.to(device)
                    label_gt = label_gt.to(device)

                    mask, label = model(image)
                    dice += [1 - dloss(mask, mask_gt).item()]
                    acc = torch.argmax(label, axis=1) == \
                        torch.argmax(label_gt, axis=1)
                    accuracy.extend(acc.cpu().numpy().ravel())
            except StopIteration:
                pass
            print('Epoch: {}/{} - accuracy: {:.4f} - dice: {:.4f}'.format(
                epoch+1, EPOCHS, np.mean(accuracy), np.mean(dice))
            )

        # save model checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'loss': loss,
                        }, SAVE_PATH)
            print('checkpoint saved.')

if __name__ == '__main__':

    # plug-in your model here
    model = UNet(channels=1, classes=1, subpixels=9).to(device)  

    criterion = combine_loss()
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=LEARNING_RATE*10,
        steps_per_epoch=STEPS,
        pct_start=0.15,
        epochs=EPOCHS
    )

    # start from last checkpoint
    if INIT_EPOCH > 0:
        checkpoint = torch.load(SAVE_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        INIT_EPOCH = checkpoint['epoch']
        # loss = checkpoint['loss']

    train(model, criterion, opt, scheduler)