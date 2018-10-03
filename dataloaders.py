import random
from os import listdir,path
from numpy import loadtxt
import numpy as np
import torch
from PIL.Image import open as imopen
from torchvision.transforms import Normalize,ColorJitter,ToTensor,functional,ToPILImage,RandomCrop,RandomRotation,RandomHorizontalFlip,RandomVerticalFlip
import torch.nn.functional as F
from functions import flip_kernels

def rand_bool(probability=0.5):
    return bool(torch.rand(1) < probability);

def pad_kernel(kernel,out_shape):
    k_shape = kernel.shape;
    pad2 = (out_shape[0]-k_shape[0])//2;
    pad4 = (out_shape[1]-k_shape[1])//2;
    pad3 = pad4 + (out_shape[1]-k_shape[1])%2;
    pad1 = pad2 + (out_shape[0]-k_shape[0])%2;
    return F.pad(kernel.unsqueeze(0).unsqueeze(0),(pad3,pad4,pad1,pad2)).squeeze(0);

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir,gamma=True):
        # init params
        self.root_dir = root_dir;
#        self.crop = crop;
        self.gamma = gamma;
        
        # loading files dictionary
        self.blurred_dict = [i for i in listdir(self.root_dir + '/blurred_imgs') if path.splitext(i)[1] == '.txt'];
        # image transforms
        self.pil2tensor = ToTensor();
    
    def __len__(self):
        return len(self.blurred_dict);

    def __getitem__(self, idx):        
        
        # processing image
        blurred = ToPILImage()(np.expand_dims(np.uint8(loadtxt(self.root_dir + '/blurred_imgs/' + self.blurred_dict[idx])*255),2));
        gt = ToPILImage()(np.expand_dims(np.uint8(loadtxt(self.root_dir + '/gt_imgs/' + 'x' + self.blurred_dict[idx][1:])*255),2));
        kern = torch.FloatTensor(loadtxt(self.root_dir + '/kernels/' + 'k_' + self.blurred_dict[idx][-5:]));
        
        if self.gamma:
            gamma = (torch.rand(1)*0.8+0.4)[0];
            blurred = functional.adjust_gamma(blurred, gamma, gain=1);
            gt = functional.adjust_gamma(gt, gamma, gain=1);
        
        if rand_bool():
            blurred = functional.hflip(blurred);
            gt = functional.hflip(gt);
            kern = kern.flip(1);
            
        blurred = self.pil2tensor(blurred);
        gt = self.pil2tensor(gt);
        stddev = (torch.rand(1)*(2/255)+(1/255))[0];
        lam = 1/stddev**2;
        blurred = torch.clamp(blurred + torch.randn(blurred.shape)*stddev,0,1);
        
        return (gt,blurred,F.pad(kern,((23-kern.shape[0])//2,(23-kern.shape[0])//2,(23-kern.shape[1])//2,(23-kern.shape[1])//2)).unsqueeze(0),lam.unsqueeze(0));
    
class TrainDataset_synth(torch.utils.data.Dataset):
    def __init__(self, root_dir,kern_augment=True,img_augment=True):
        # init params
        self.root_dir = root_dir;
#        self.crop = crop;
        self.img_augment = img_augment;
        self.k_augment = kern_augment;
        
        # loading files dictionary
        self.gt_dict = [i for i in listdir(self.root_dir + '/gt_imgs') if path.splitext(i)[1] == '.png'];
        
        # image & kernel transforms
        self.pil2tensor = ToTensor();
        self.crop = RandomCrop((277,277));
        self.rotate = RandomRotation(180);
        self.hflip = RandomHorizontalFlip(p=0.5);
        self.vflip = RandomVerticalFlip(p=0.5);
        
    def __len__(self):
        return len(self.gt_dict)*8;

    def __getitem__(self, idx):        
        
        # processing image
        image = (idx)//8; # from 1 to 80
        kern = idx//len(self.gt_dict) + 1; # from 1 to 8
        
        gt = self.crop(imopen(self.root_dir + '/gt_imgs/' + self.gt_dict[image]));
        kern = imopen(self.root_dir + '/kernels/' + 'kernel' + str(kern) + '_groundtruth_kernel.png');

        if self.img_augment:
            gamma = (torch.rand(1)*0.8+0.4)[0];
            gt = functional.adjust_gamma(gt, gamma, gain=1);
            if rand_bool():
                gt = functional.hflip(gt);
        gt = self.pil2tensor(gt);
        
        if self.k_augment:
            kern = self.rotate(kern);
            kern = self.hflip(kern);
            kern = self.vflip(kern);
        kern = self.pil2tensor(kern).squeeze(0);
        kern = kern/torch.sum(kern);
        kern = F.pad(kern,((23-kern.shape[0])//2,(23-kern.shape[0])//2,(23-kern.shape[1])//2,(23-kern.shape[1])//2)).unsqueeze(0);
        
        blurred = F.conv2d(gt.unsqueeze(0),flip_kernels(kern.unsqueeze(0))).squeeze(0);
        gt = gt[:,11:-11,11:-11];
        
        stddev = (torch.rand(1)*(2/255)+(1/255))[0];
        lam = 1/stddev**2;
        blurred = torch.clamp(blurred + torch.randn(blurred.shape)*stddev,0,1);
        
        return (gt,blurred,kern,lam.unsqueeze(0));
    
class Dataset_color(torch.utils.data.Dataset):
    def __init__(self, root_dir,kern_augment=False,img_augment=False,img_size=(255,255),train=True):
        # init params
        self.root_dir = root_dir;
#        self.crop = crop;
        self.img_augment = img_augment;
        self.k_augment = kern_augment;
        
        self.img_size = img_size;
        
        if train:
            self.subfolder = '/train/';
        else:
            self.subfolder = '/test/';
            
        # loading files dictionary
        self.gt_dict = [i for i in listdir(self.root_dir + '/color_imgs/' + self.subfolder + '/gt_images/') if path.splitext(i)[1] == '.png'];
        self.kern_dict = [i for i in listdir(self.root_dir + '/kerns/' + self.subfolder) if path.splitext(i)[1] == '.txt'];
        
        # image & kernel transforms
        self.pil2tensor = ToTensor();
        
        self.color = ColorJitter(brightness=0.1,contrast=0.3,saturation=0.5,hue=0.03);
        self.rotate = RandomRotation(180);
        self.hflip = RandomHorizontalFlip(p=0.5);
        self.vflip = RandomVerticalFlip(p=0.5);
        
        self.take_random = random.SystemRandom();
        
    def __len__(self):
        return len(self.gt_dict)*10;

    def __getitem__(self, idx):        
        
        # processing image
        image = (idx)//10;
        
        kern = torch.FloatTensor(np.loadtxt(self.root_dir + '/kerns/' + self.subfolder + self.take_random.choice(self.kern_dict)));
        k_shape = kern.shape;
        gt = RandomCrop((self.img_size[0]+k_shape[0]-1,self.img_size[1]+k_shape[1]-1))(imopen(self.root_dir + '/color_imgs/' + self.subfolder + '/gt_images/' + self.gt_dict[image]));
        
        if self.img_augment:
            gt = self.color(gt);
            if rand_bool():
                gt = functional.hflip(gt);
        gt = self.pil2tensor(gt);
        
        if self.k_augment:
            kern = ToPILImage()(kern);
            kern = self.rotate(kern);
            kern = self.hflip(kern);
            kern = self.vflip(kern);
            kern = self.pil2tensor(kern).squeeze(0);
            
        kern = kern/torch.sum(kern);
        blurred = F.conv2d(gt.unsqueeze(0),flip_kernels(kern.unsqueeze(0).unsqueeze(0)).expand(3,1,k_shape[0],k_shape[1]),groups=3).squeeze(0);
        
        gt = gt[:,(k_shape[0]-1)//2:-(k_shape[0]-1)//2,(k_shape[1]-1)//2:-(k_shape[1]-1)//2];
        
        kern = pad_kernel(kern,(81,81));
        
        stddev = (torch.rand(1)*(2/255)+(1/255))[0];
        lam = 1/stddev**2;
        blurred = torch.clamp(blurred + torch.randn(blurred.shape)*stddev,0,1);
        
        return (gt,blurred,kern,lam.unsqueeze(0));
    
class Dataset_color_wiener(torch.utils.data.Dataset):
    def __init__(self, root_dir,kern_augment=False,img_augment=False,img_size=(255,255),train=True,noise=3):
        # init params
        self.root_dir = root_dir;
#        self.crop = crop;
        self.img_augment = img_augment;
        self.k_augment = kern_augment;
        
        self.img_size = img_size;
        self.noise = noise;
        
        if train:
            self.subfolder = '/train/';
        else:
            self.subfolder = '/test/';
            
        # loading files dictionary
        self.gt_dict = [i for i in listdir(self.root_dir + '/color_imgs/' + self.subfolder + '/gt_images/') if path.splitext(i)[1] == '.png'];
        self.kern_dict = [i for i in listdir(self.root_dir + '/kerns/' + self.subfolder) if path.splitext(i)[1] == '.txt'];
        
        # image & kernel transforms
        self.pil2tensor = ToTensor();
        
        self.color = ColorJitter(brightness=0.1,contrast=0.3,saturation=0.5,hue=0.03);
        self.rotate = RandomRotation(180);
        self.hflip = RandomHorizontalFlip(p=0.5);
        self.vflip = RandomVerticalFlip(p=0.5);
        
        self.take_random = random.SystemRandom();
        
    def __len__(self):
        return len(self.gt_dict)*10;

    def __getitem__(self, idx):        
        
        # processing image
        image = (idx)//10;
        
        kern = torch.FloatTensor(np.loadtxt(self.root_dir + '/kerns/' + self.subfolder + self.take_random.choice(self.kern_dict)));
        k_shape = kern.shape;
        gt = RandomCrop((self.img_size[0]+k_shape[0]-1,self.img_size[1]+k_shape[1]-1))(imopen(self.root_dir + '/color_imgs/' + self.subfolder + '/gt_images/' + self.gt_dict[image]));
        
        if self.img_augment:
            gt = self.color(gt);
            if rand_bool():
                gt = functional.hflip(gt);
        gt = self.pil2tensor(gt);
        
        if self.k_augment:
            kern = ToPILImage()(kern);
            kern = self.rotate(kern);
            kern = self.hflip(kern);
            kern = self.vflip(kern);
            kern = self.pil2tensor(kern).squeeze(0);
            
        kern = kern/torch.sum(kern);
        blurred = F.conv2d(gt.unsqueeze(0),flip_kernels(kern.unsqueeze(0).unsqueeze(0)).expand(3,1,k_shape[0],k_shape[1]),groups=3).squeeze(0);
        
        gt = gt[:,(k_shape[0]-1)//2:-(k_shape[0]-1)//2,(k_shape[1]-1)//2:-(k_shape[1]-1)//2];
        
        kern = pad_kernel(kern,(81,81));
        
        #stddev = (torch.rand(1)*(2/255)+(1/255))[0];
        #lam = 1/stddev**2;
        blurred = torch.clamp(blurred + torch.randn(blurred.shape)*(self.noise/255),0,1);
        
        return (gt,blurred,kern);