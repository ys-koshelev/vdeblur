import torch
import torch.nn as nn
from torchvision.models.densenet import DenseNet
from functions import edgetaper_torch,pad_for_kernel_torch,adjust_bounds,psf2otf_torch,otf2psf_torch
from Ctorch import Cinit,Cabs,Cconj,Cmul,Creal
from torch.autograd import Function

class FourierDeconvolutionFunction_init(Function):
    @staticmethod
    def forward(ctx, y, k, w, phi, weights):
        weights_shape = torch.ShortTensor(list(weights.shape))
        
        Fy = torch.rfft(y, 2, onesided=False)
        Dg = psf2otf_torch(weights, Fy.shape[2:4])
        Dk = psf2otf_torch(k, Fy.shape[2:4]).expand(Fy.shape)
        Fphi = torch.rfft(phi/w[:,:,None,None], 2, onesided=False)
        
        Dg_sum = (torch.sum(Cabs(Dg)**2, dim=0).unsqueeze(0)/w[:,:,None,None]).expand(Dk.shape[:-1])
        Dg = Dg.expand(Dg.shape[:1] + Fy.shape[1:])
        
        Omega = Cabs(Dk)**2 + Dg_sum
        Omega = 1/Omega
        
        num = Omega[...,None]*(Cmul(Cconj(Dk), Fy) + Fphi)
        
        ctx.save_for_backward(Omega, Omega[...,None]*num, Fphi, Dg_sum, w, Dg, weights_shape)
    
        output = torch.irfft(num, 2, onesided=False)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        
        Omega, num, Fphi, Dg_sum, w, Dg, weights_shape = ctx.saved_variables
        grad_w = grad_phi = grad_weight = None
        Fgrad = torch.rfft(grad_output, 2, onesided=False)
        
        if ctx.needs_input_grad[2]:
            grad_w = -torch.sum(torch.irfft(
                ((Omega[...,None])*Fphi - (Dg_sum[...,None])*num), 2, onesided=False)*grad_output/w[:,:,None,None], dim=(1,2,3)).unsqueeze(-1)
        if ctx.needs_input_grad[3]:
            grad_phi = (torch.irfft(Omega[...,None]*Fgrad, 2, onesided=False))/w[:,:,None,None]
            
            
        if ctx.needs_input_grad[4]:
            grad_weight = Dg.new(*weights_shape);
            temp = Creal(Cmul(num, Cconj(Fgrad)));
            
            for i in range(weights_shape[0]):
                 grad_weight[i,...] = -2*torch.sum(
                     otf2psf_torch((Dg[i,...])[None,...]*temp[...,None], tuple(weights_shape[2:4]))/w[:,:,None,None], dim=0)
        
        return None, None, grad_w, grad_phi, grad_weight

class FourierDeconvolution(nn.Module):
    def __init__(self, stage=1,n_filters=24,filter_size=5, grayscale=False):
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
        else:
            y,x,k,lam = inputs;
        
        w = self.omega(lam);
        cnn_res = self.cnn(x);
        
        return FourierDeconvolutionFunction_init.apply(x, k, w, cnn_res, self.filters)