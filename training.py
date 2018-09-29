import torch
import torch.nn as nn
import shutil
from tensorboardX import SummaryWriter
import os
import numpy as np
from tqdm import tnrange,tqdm_notebook
from torch.utils.data import DataLoader
from time import time
from dataloaders import Dataset_color, Dataset_color_wiener
from visualise import visualise_weights

# compute loss
def compute_loss_cuda(model,x,y,k,lam):
    batch_net = model((y.cuda(),k.cuda(),lam.cuda()));
    loss = -10*torch.log10((1/nn.functional.mse_loss(batch_net[:,:,(k.shape[2]-1)//2:-(k.shape[2]-1)//2,(k.shape[3]-1)//2:-(k.shape[3]-1)//2],x.cuda())));
    #loss = nn.functional.mse_loss(batch_net[:,:,11:-11,11:-11],x.cuda());
    return loss;

def compute_loss_wiener_cuda(model,x,y,k):
    batch_net = model((y.cuda(),k.cuda()));
    loss = -10*torch.log10((1/nn.functional.mse_loss(batch_net[:,:,(k.shape[2]-1)//2:-(k.shape[2]-1)//2,(k.shape[3]-1)//2:-(k.shape[3]-1)//2],x.cuda())));
    #loss = nn.functional.mse_loss(batch_net[:,:,11:-11,11:-11],x.cuda());
    return loss;

def are_nans(net):
    """
        Function for checking network weight for NaNs
        
        Input:
        
            - net[torch.nn.Module]: network which weights to check
        
        Output:
            - [bool]: True - there are NaNs, False - there is no NaNs in weights
    """
    u = 0;
    for i in range(len(list(net.parameters()))):
        u = u + torch.sum(torch.isnan(list(net.parameters())[i]));
    return u!=0;

def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    """
        Function for saving current training state checkpoint
        
        Input:
        
            - state[dict]: dict with network and optimizer weights
            
            - is_best[bool]: for saving checkpoint with best performance
        
        Output: None
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pt')
    pass

def train(model,num_epochs,directory,b_size,dictname='FNet',pretrained=None,gpu=0):
    """
        Function for training.
        
        Input:
            - model[torch.nn.Module]: network for loss computation
            
            - num_epochs[int]: number of epochs to perform
            
            - directory[string]: directory to dataset
            
            - b_size[int]: minibatch size during training
            
            - dictname[string]: name of folders and tensorboard tags corresponding to this training routine
            
            - pretrained[string]: path to pretrained checkpoint to coninue training
            
            - gpu[int]: ID of GPU to use in training
            
        Output: None
    """
    is_best = False;
    maxpsnr = 0;
    
    # creating folders
    if not(os.path.isdir('tboard/' + dictname)):
        os.mkdir(('tboard/' + dictname));
    if not(os.path.isdir('trained/' + dictname)):
        os.mkdir(('trained/' + dictname));
    writer = SummaryWriter(log_dir=('tboard/' + dictname));
    
    # loading dataset for training
    trainset = Dataset_color(directory,img_size=(500,500),train=True);
    trainloader = DataLoader(trainset, batch_size=b_size, shuffle=True, num_workers=6);
    
    valset = Dataset_color(directory,img_size=(500,500),train=False);
    valloader = DataLoader(trainset, batch_size=b_size, shuffle=False, num_workers=6);
    
    # optimizer
    opt = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=1e-6,amsgrad=True);
    
    model.cuda()
    
    # loading checkpoint
    start_epoch = 0;
    if pretrained:
        if os.path.isfile(pretrained):
            print("=> loading checkpoint '{}'".format(pretrained))
            checkpoint = torch.load(pretrained,map_location=lambda storage, loc: storage.cuda(gpu));
            start_epoch = checkpoint['epoch'] + 1;
            model.load_state_dict(checkpoint['state_dict']);
            opt.load_state_dict(checkpoint['optimizer']);
            print("=> loaded checkpoint '{}' (epoch {})".format(pretrained, checkpoint['epoch']));
            del checkpoint;
        else:
            print("=> no checkpoint found at '{}'".format(pretrained));
    torch.cuda.empty_cache();
    # Full pass over the training data
    for epoch in tnrange(start_epoch,num_epochs+start_epoch):
        
        torch.manual_seed(6838019176185359526 - epoch);
        np.random.seed(1431655765 - epoch);
        torch.cuda.manual_seed_all(6838019176185359526 - epoch);
        
        # number of passes in 1 epoch
        l = len(trainloader);
        
        # timings
        times = time();
        
        # Training network mode for dropouts and batchnorms
        model.train(True);
        # Train on batch
        for i_batch, (x,y,k,l) in enumerate((tqdm_notebook(trainloader))):
            # obtaining loss
            loss = compute_loss_cuda(model,x,y,k,l);
            
            # writing to TensorBoard
            writer.add_scalar('Training loss', loss.data.cpu(), (epoch*len(trainloader) + i_batch)*trainloader.batch_size);
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        model.train(False);
        valloss = [];
        for i_batch, (x,y,k,l) in enumerate((valloader)):
            # obtaining loss
            valloss.append(compute_loss_cuda(model,x,y,k,l).detach().cpu());
        valloss = -np.mean(valloss);
        # writing to TensorBoard
        writer.add_scalar('Validation loss', valloss, epoch);
        
        # writing to TensorBoard
        writer.add_scalar('Training time', time() - times, epoch);
        writer.add_image('Learnable filters', visualise_weights(model.state_dict()['filters'].detach().cpu()), epoch);
        writer.add_image('Init convolution weights', visualise_weights(model.cnn.conv0.weight.detach().cpu()), epoch);
        
        if are_nans(model):
            print('NaNs:', 'NaNs detected in weights!', epoch)
            #writer.add_text('NaNs:', 'NaNs detected in weights!', epoch)
            break
    
        if valloss > maxpsnr:
            maxpsnr = valloss;
            is_best = True;
            
        save_checkpoint({
                'epoch': epoch,
                'arch': 'FDN',
                'state_dict': model.state_dict(),
                'optimizer' : opt.state_dict(),
                }, is_best, filename=('trained/' + dictname + '/' + 'FDN' + '_epoch_' + str(epoch) + '.pt'));
        is_best = False;
    
    writer.close();
    pass
'''
        # is best in terms of validation loss?
        if valloss < minloss:
            minloss = valloss;
            is_best = True;

        
'''

def train_wiener(model,num_epochs,directory,b_size,dictname='FNet',pretrained=None,gpu=0):
    """
        Function for training.
        
        Input:
            - model[torch.nn.Module]: network for loss computation
            
            - num_epochs[int]: number of epochs to perform
            
            - directory[string]: directory to dataset
            
            - b_size[int]: minibatch size during training
            
            - dictname[string]: name of folders and tensorboard tags corresponding to this training routine
            
            - pretrained[string]: path to pretrained checkpoint to coninue training
            
            - gpu[int]: ID of GPU to use in training
            
        Output: None
    """
    is_best = False;
    maxpsnr = 0;
    
    # creating folders
    if not(os.path.isdir('tboard/' + dictname)):
        os.mkdir(('tboard/' + dictname));
    if not(os.path.isdir('trained/' + dictname)):
        os.mkdir(('trained/' + dictname));
    writer = SummaryWriter(log_dir=('tboard/' + dictname));
    
    # loading dataset for training
    trainset = Dataset_color_wiener(directory,img_size=(500,500),train=True);
    trainloader = DataLoader(trainset, batch_size=b_size, shuffle=True, num_workers=12);
    
    valset = Dataset_color_wiener(directory,img_size=(500,500),train=False);
    valloader = DataLoader(trainset, batch_size=b_size, shuffle=False, num_workers=12);
    
    # optimizer
    opt = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=1e-6,amsgrad=True);
    
    model.cuda()
    
    # loading checkpoint
    start_epoch = 0;
    if pretrained:
        if os.path.isfile(pretrained):
            print("=> loading checkpoint '{}'".format(pretrained))
            checkpoint = torch.load(pretrained,map_location=lambda storage, loc: storage.cuda(gpu));
            start_epoch = checkpoint['epoch'] + 1;
            model.load_state_dict(checkpoint['state_dict']);
            opt.load_state_dict(checkpoint['optimizer']);
            print("=> loaded checkpoint '{}' (epoch {})".format(pretrained, checkpoint['epoch']));
            del checkpoint;
        else:
            print("=> no checkpoint found at '{}'".format(pretrained));
    torch.cuda.empty_cache();
    # Full pass over the training data
    for epoch in tnrange(start_epoch,num_epochs+start_epoch):
        
        torch.manual_seed(6838019176185359526 - epoch);
        np.random.seed(1431655765 - epoch);
        torch.cuda.manual_seed_all(6838019176185359526 - epoch);
        
        # number of passes in 1 epoch
        l = len(trainloader);
        
        # timings
        times = time();
        
        # Training network mode for dropouts and batchnorms
        model.train(True);
        # Train on batch
        for i_batch, (x,y,k) in enumerate((tqdm_notebook(trainloader))):
            # obtaining loss
            loss = compute_loss_wiener_cuda(model,x,y,k);
            
            # writing to TensorBoard
            writer.add_scalar('Training loss', loss.data.cpu(), (epoch*len(trainloader) + i_batch)*trainloader.batch_size);
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        model.train(False);
        valloss = [];
        for i_batch, (x,y,k) in enumerate((valloader)):
            # obtaining loss
            valloss.append(compute_loss_wiener_cuda(model,x,y,k).detach().cpu());
        valloss = -np.mean(valloss);
        # writing to TensorBoard
        writer.add_scalar('Validation loss', valloss, epoch);
        
        # writing to TensorBoard
        writer.add_scalar('Training time', time() - times, epoch);
        writer.add_image('Init convolution weights', visualise_weights(model.filters.detach().cpu(),nv=4,nh=2), epoch);
        
        if are_nans(model):
            print('NaNs:', 'NaNs detected in weights!', epoch)
            #writer.add_text('NaNs:', 'NaNs detected in weights!', epoch)
            break
    
        if valloss > maxpsnr:
            maxpsnr = valloss;
            is_best = True;
            
        save_checkpoint({
                'epoch': epoch,
                'arch': 'FDN',
                'state_dict': model.state_dict(),
                'optimizer' : opt.state_dict(),
                }, is_best, filename=('trained/' + dictname + '/' + 'FDN' + '_epoch_' + str(epoch) + '.pt'));
        is_best = False;
    
    writer.close();
    pass