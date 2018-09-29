import torch
import torch.nn as nn
from torchvision.models.densenet import DenseNet
from functions import edgetaper_torch,pad_for_kernel_torch,adjust_bounds,psf2otf_torch
from Ctorch import Cinit,Cabs,Cconj

class FourierDeconvolution(nn.Module):
    def __init__(self, stage,n_filters=24,filter_size=5, grayscale=False):
        super(FourierDeconvolution, self).__init__()
        
        self.stage = stage
        self.grayscale = grayscale;
        
        self.omega = nn.Sequential(nn.Linear(1,16,bias=False),
                                   nn.BatchNorm1d(16),
                                   nn.ELU(),
                                   nn.Linear(16,16,bias=False),
                                   nn.BatchNorm1d(16),
                                   nn.ELU(),
                                   nn.Linear(16,1),
                                   nn.Softplus()
                                  );
        
        self.cnn = DenseNet(num_init_features=32,block_config=(2,)).features;
        self.cnn.add_module('reluf',nn.ReLU());
        self.cnn.pool0 = nn.Conv2d(32,32,kernel_size=(3,3), stride=1, padding=(1,1), bias=False);
        if grayscale:
            self.cnn.conv0 = nn.Conv2d(1,32,kernel_size=(5,5), stride=1, padding=(2,2), bias=False);
            self.cnn.add_module('convf',nn.Conv2d(96,1,kernel_size=(3,3),padding=(1,1)));
            self.filters = nn.Parameter(nn.init.kaiming_normal_(torch.zeros(n_filters,1,filter_size,filter_size)));
        else:
            self.cnn.conv0 = nn.Conv2d(3,32,kernel_size=(5,5), stride=1, padding=(2,2), bias=False);
            self.cnn.add_module('convf',nn.Conv2d(96,3,kernel_size=(3,3),padding=(1,1)));
            self.filters = nn.Parameter(nn.init.kaiming_normal_(torch.zeros(n_filters,3,filter_size,filter_size)));
        
    def forward(self,inputs):
        if self.stage == 1:
            y, k, lam = inputs;
            x = edgetaper_torch(pad_for_kernel_torch(y,k),k);
            phi = torch.rfft(x,2,onesided=False);
        else:
            y,x,k,lam = inputs;
            phi = torch.rfft(adjust_bounds(x,y,k),2,onesided=False); 
        
        filters_otf = psf2otf_torch(self.filters,phi.shape[-3:-1]);
        k_otf = psf2otf_torch(k,phi.shape[-3:-1]);
        if not(self.grayscale):
            k_otf = k_otf.expand(phi.shape);
        
        w = self.omega(lam);
        cnn_res = self.cnn(x)/w[:,:,None,None];
        
        filters_otf = torch.sum(Cabs(filters_otf)**2,dim=0).unsqueeze(0).expand(k_otf.shape[:-1])/w[:,:,None,None];
        
        return torch.irfft(torch.rfft(torch.irfft(Cconj(k_otf)*phi,2,onesided=False)+cnn_res,
                   2,onesided=False)/((Cabs(k_otf)**2 + filters_otf)[...,None]),2,onesided=False);