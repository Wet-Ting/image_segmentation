import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.autograd import Function
import torch.nn.functional as F

from torch import optim
from UNet import UNet_ver2
from dataloader import myDataset

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

device = torch.device("cuda")

img1 = 'data/f01/image/'
mask1 = 'data/f01/label/'

img2 = 'data/f02/image/'
mask2 = 'data/f02/label/'

img3 = 'data/f03/image/'
mask3 = 'data/f03/label/'

class DC(Function):
    def forward(self, input, target):
    
        self.save_for_backward(input, target)
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target)

        return (2 * self.inter.float()) / self.union.float()
        
    def backward(self, output):
    
        input, target = self.saved_variables
        grad_input = grad_target = None
        
        if self.needs_input_grad[0]:
            grad_input = output * 2 * (target * self.union - self.inter) / (self.union * self. union)
            
        if self.needs_input_grad[1]:
            grad_target = None
            
        return grad_input, grad_target
        
def dice_coeff(input, target):

    s = torch.FloatTensor(1).cuda().zero_()
    
    for i, c in enumerate(zip(input, target)):
        s += DC().forward(c[0], c[1])
        
    return s / (i + 1)
    
def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim = 2).sum(dim = 2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim = 2).sum(dim = 2) + target.sum(dim = 2).sum(dim = 2) + smooth)))
    
    return loss.mean()
    
def calc_loss(pred, target, bce_weight = 0):
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    return loss

rgb2grayWeights = [0.2989, 0.5870, 0.1140]

def train(net, loader, criterion, optimizer):
    net.train()
    epoch_loss = 0
    
    for batch in loader:
        imgs = batch['image']
        masks = batch['mask']        
        
        if masks.size(1) == 3:
            masks = rgb2grayWeights[0] * masks[:, 0, :, :] + rgb2grayWeights[1] * masks[:, 1, :, :] + \
                       rgb2grayWeights[2] * masks[:, 2, :, :]
            masks.unsqueeze_(1)
        
        imgs = imgs.to(device = device, dtype = torch.float32)
        mask_type = torch.float32 if net.n_classes == 1 else torch.long
        masks = masks.to(device = device, dtype = mask_type)
        
        masks_pred = net(imgs)
        
        # result = Image.fromarray((masks_pred * 255).astype(np.uint8))
        # result.save(out_files)
        img = np.squeeze(imgs.cpu().detach().numpy())
        # mask_pred = np.squeeze(masks_pred.cpu().detach().numpy())
        # plot_img_and_mask(img, mask_pred)
        
        loss = calc_loss(masks_pred, masks)
        epoch_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return epoch_loss

def trainNet(net, epochs = 120, batch_size = 1, lr = 0.0001, save = True, img_scale = 1):

    dataset1 = myDataset(img1, mask1, img_scale)
    dataset2 = myDataset(img2, mask2, img_scale)
    dataset3 = myDataset(img3, mask3, img_scale)
    
    loader1 = DataLoader(dataset1, batch_size = batch_size, shuffle = True, num_workers = 2, pin_memory = True)
    loader2 = DataLoader(dataset2, batch_size = batch_size, shuffle = True, num_workers = 2, pin_memory = True)
    loader3 = DataLoader(dataset3, batch_size = batch_size, shuffle = True, num_workers = 2, pin_memory = True)
    
    optimizer = optim.Adam(net.parameters(), lr = lr, betas = (0.5, 0.999))
    # optimizer = optim.SGD(net.parameters(), lr = lr)
    
    
    criterion = nn.BCEWithLogitsLoss()
    # criterion = DiceLoss()
    
    best_dc1 = best_dc2 = best_dc3 = 0
    loss1 = []
    loss2 = []
    loss3 = []
    dc1_list = []
    dc2_list = []
    dc3_list = []
    
    for epoch in range(epochs):
        print('epoch: ' + str(epoch + 1) + '/' + str(epochs))
 
        # k-fold validation 
        l1 = l2 = l3 = dc1 = 0 
        # epoch_loss = train(net, loader1, criterion, optimizer)
        l1 = train(net, loader1, criterion, optimizer)
        l2 = train(net, loader2, criterion, optimizer)
        dc1 = eval(net, loader3)
        print('1st: ' + str((l1 + l2) / 40))
        print('valid dc: ' + str(dc1))   
        print('------------------')
        loss1.append((l1 + l2) / 40)
        dc1_list.append(dc1)
        
        if save:
            torch.save(net.state_dict(), 'ckpt_03_ver6/' + f'epoch{epoch + 1}.pth')
        
        if dc1 > best_dc1 :
            torch.save(net.state_dict(), 'ckpt_03_ver6/' + 'best.pth')
        
        l1 = l2 = dc2 = 0
        l1 = train(net, loader1, criterion, optimizer)
        l3 = train(net, loader3, criterion, optimizer)
        dc2 = eval(net, loader2)       
        print('2nd: ' + str((l1 + l3) / 40))
        print('valid dc: ' + str(dc2)) 
        print('------------------')
        loss2.append((l1 + l3) / 40)
        dc2_list.append(dc2)
        
        if save:
            torch.save(net.state_dict(), 'ckpt_02_ver6/' + f'epoch{epoch + 1}.pth')
        
        if dc2 > best_dc2 :
            torch.save(net.state_dict(), 'ckpt_02_ver6/' + 'best.pth')
        
        l1 = l3 = dc3 = 0
        l2 = train(net, loader2, criterion, optimizer)
        l3 = train(net, loader3, criterion, optimizer)
        dc3 = eval(net, loader1)       
        print('3rd: ' + str((l2 + l3) / 40))
        print('valid dc: ' + str(dc3)) 
        print('------------------')
        loss3.append((l2 + l3) / 40)
        dc3_list.append(dc3)
        
        if save:
            torch.save(net.state_dict(), 'ckpt_01_ver6/' + f'epoch{epoch + 1}.pth')
        
        if dc3 > best_dc3 :
            torch.save(net.state_dict(), 'ckpt_01_ver6/' + 'best.pth')
            
        plt.figure()
        plt.plot(loss1)
        plt.plot(loss2)
        plt.plot(loss3)
        plt.title('loss')
        plt.ylabel('loss'), plt.xlabel('epoch')
        plt.legend(['valid_3', 'valid_2', 'valid_1'], loc = 'upper left')
        #plt.show()
        plt.savefig('loss.png')
        plt.close()
    
        plt.figure()
        plt.plot(dc1_list)
        plt.plot(dc2_list)
        plt.plot(dc3_list)
        plt.title('dc acc')
        plt.ylabel('acc'), plt.xlabel('epoch')
        plt.legend(['valid_3', 'valid_2', 'valid_1'], loc = 'upper left')
        #plt.show()
        plt.savefig('valid_dc_acc.png')
        plt.close()
            
def eval(net, loader):
    net.eval()
    total = 0
    
    for batch in loader:
        imgs = batch['image']
        masks = batch['mask']
        
        imgs = imgs.to(device = device, dtype = torch.float32)
        mask_type = torch.float32 if net.n_classes == 1 else torch.long
        masks = masks.to(device = device, dtype = mask_type)
        
        mask_pred = net(imgs)
        
        for gt, pred in zip(masks, mask_pred):
            pred = (pred > 0.5).float()
            total += dice_coeff(pred, gt.squeeze(dim = 1)).item()
            # print(dice_coeff(pred, gt.squeeze(dim = 1)).item())
            
    return total / 20
            
if __name__ == '__main__':
    # net = UNet(n_channels = 3, n_classes = 1)
    net = UNet_ver2(n_classes = 1)
    net.to(device)
    trainNet(net)