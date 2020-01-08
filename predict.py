import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.autograd import Function
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import cv2 as cv
from skimage import morphology

from torch import optim

from UNet import UNet_ver2
from dataloader import myDataset

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def predict_img(net, full_img, scale_factor = 1, out_threshold = 0.95):
    
    net.eval()
    
    img = myDataset.preprocess(full_img, scale_factor)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0).to(device = device, dtype = torch.float32)
    
    with torch.no_grad():
    
        output = net(img)
        probs = torch.sigmoid(output)
        probs = probs.squeeze(0)
        
        shape = np.array(full_img)
        
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(shape.shape[1]),
            transforms.ToTensor()
        ])
        
        probs = tf(probs.cpu())       
        mask = probs.squeeze().cpu().numpy()
                   
    return mask > out_threshold
                
def plot_img_and_mask(img, gt, mask, out_files):

    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 3)
    img = cv.resize(img, (500, 1166))
    ax[0].set_title('Input image')
    ax[0].imshow(img, 'gray')          
    
    mask = morphology.remove_small_objects(mask, connectivity=1, in_place=False, min_size=1000)
    
    r = Image.fromarray((mask * 255).astype(np.uint8))
    r.save(out_files)
    
    # T/F -> 1/0
    result = mask + 0
    result = np.clip(result, 0, 255)
    result = np.array(result, np.uint8)

    image, contours, _ = cv.findContours(result, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.drawContours(img, contours, -1, (255, 0, 0), 2) 
    
    ax[1].set_title('Ground truth')
    ax[1].imshow(gt, 'gray')

    ax[2].set_title('Output mask')
    ax[2].imshow(mask, 'gray')
    
    ax[3].set_title('Predict')
    ax[3].imshow(img)
    
    plt.xticks([]), plt.yticks([])
    plt.show()
    
def calculate_each(gt, mask, out_files):
  
    gt = cv.resize(gt, (500, 1166))
    gt_ = gt
    gt = cv.cvtColor(gt, cv.COLOR_BGR2GRAY)
    
    m = cv.imread(out_files)
    
    mask = morphology.remove_small_objects(mask, connectivity = 1, in_place = False, min_size = 1000)
    
    # T/F -> 1/0
    result = mask + 0
    result_ = np.clip(result, 0, 255)
    result_ = np.array(result, np.uint8)
         
    _, contours_gt, _ = cv.findContours(gt, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)       
    _, contours_mask, _ = cv.findContours(result_, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    j = k = dc_sum = 0
    
    for i in range(len(contours_mask)):
    
        print('gt_' + str(i+1) + ': ')
        print(cv.contourArea(contours_gt[i]))
        print('mask_' + str(i+1) + ': ')
        print(cv.contourArea(contours_mask[i]))
        
        each_gt = np.zeros(result_.shape, dtype = 'uint8')
        each_mask = np.zeros(result_.shape, dtype = 'uint8')
                      
        if cv.contourArea(contours_mask[i]) > (cv.contourArea(contours_gt[i]) * 1.5):
            cv.drawContours(each_gt, contours_gt[j : j + 2], -1, 255, -1)            
            cv.drawContours(each_mask, contours_mask[k : k + 1], -1, 255, -1)
            j += 2
            k += 1           
        elif cv.contourArea(contours_gt[i]) > (cv.contourArea(contours_mask[i]) * 1.5):
            cv.drawContours(each_gt, contours_gt[j : j + 1], -1, 255, -1)
            cv.drawContours(each_mask, contours_mask[k : k + 2], -1, 255, -1)
            j += 1
            k += 2
        else:
            cv.drawContours(each_gt, contours_gt[j : j + 1], -1, 255, -1)
            cv.drawContours(each_mask, contours_mask[i : i + 1], -1, 255, -1)
            j += 1
            k += 1
        
        # cv.imshow('each_gt', each_gt)                
        # cv.imshow('each_mask', each_mask)
        # cv.waitKey(0)
               
        dc = dice(each_gt, each_mask)
        print('DC' + str(j) + ': ' + str(dc))       
        print('--------------')
        
        if(j == len(contours_mask) or k == len(contours_mask)):
            break                  
            
    dc_mean = dice(gt, result_)
    print('dc_mean: ' + str(dc_mean))
              
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dice(true_mask, pred_mask, non_seg_score=1.0):

    assert true_mask.shape == pred_mask.shape

    true_mask = np.asarray(true_mask).astype(np.bool)
    pred_mask = np.asarray(pred_mask).astype(np.bool)

    # If both segmentations are all zero, the dice will be 1. (Developer decision)
    im_sum = true_mask.sum() + pred_mask.sum()
    if im_sum == 0:
        return non_seg_score

    # Compute Dice coefficient
    intersection = np.logical_and(true_mask, pred_mask)
    return 2. * intersection.sum() / im_sum

if __name__ == '__main__':
    
    img_file = '0055.png'
    in_files = 'data/f03/image/' + img_file
    gt = 'data/f03/label/' + img_file
    out_files = 'output/' + img_file
   
    net = UNet_ver2(1).to(device) 
    net.load_state_dict(torch.load('ckpt_03_ver5/best.pth'))
    
    img = Image.open(in_files) 
    img_cv = cv.imread(in_files)
    gt_ = Image.open(gt)
    gt_cv = cv.imread(gt)
        
    mask = predict_img(net, img, 1, 0.5)    
      
    plot_img_and_mask(img_cv, gt_cv, mask, out_files)
    calculate_each(gt_cv, mask, out_files)
    