import torch
from torch.autograd import Function
from functions import psf2otf_torch, otf2psf_torch
from Ctorch import Cmul, Cabs, Cconj, Creal
import torch.nn as nn
from functions import edgetaper_torch, pad_for_kernel_torch

class WienerFunction(Function):
    @staticmethod
    def forward(ctx, y, k, lam, weights):
        
        weights_shape = torch.ShortTensor(list(weights.shape));
        Fy = torch.rfft(y, 2, onesided=False);
        Dg = psf2otf_torch(weights, Fy.shape[2:4]);
        
        Dk = psf2otf_torch(k, Fy.shape[2:4]);
        
        Dk = Dk.expand(Fy.shape);
        
        exp_lam = torch.exp(lam);
        
        Dg_sum = (torch.sum(Cabs(Dg)**2, dim=0).unsqueeze(0)*exp_lam).expand(Dk.shape[:-1]);
        Dg = Dg.expand(Dg.shape[:1] + Fy.shape[1:]);
        
        Omega = Cabs(Dk)**2 + Dg_sum;
        Omega = 1/Omega;
        
        ctx.save_for_backward(Fy, Dk, Dg_sum, Omega, Dg*exp_lam, weights_shape);
        
        output = torch.irfft(Omega[...,None]*Cmul(Cconj(Dk),Fy), 2, onesided=False);
        return output

    @staticmethod
    def backward(ctx, grad_output):
        
        Fy, Dk, Dg_sum, Omega, Dg, weights_shape = ctx.saved_variables
        grad_input = grad_lam = grad_weight = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.irfft(Omega[...,None]*Cmul(Dk, torch.rfft(grad_output, 2, onesided=False)), 2, onesided=False);
        
        if ctx.needs_input_grad[2]:
            grad_lam = -torch.sum(torch.irfft(((Omega**2)*Dg_sum)[...,None]*Cmul(Cconj(Dk), Fy), 2, onesided=False)*grad_output);
            
        if ctx.needs_input_grad[3]:
            grad_weight = Dg.new(*weights_shape);
            for i in range(weights_shape[0]):
                 grad_weight[i,...] = -2*torch.sum(
                     otf2psf_torch((Dg[i,...])[None,...]*(Creal((Omega**2)[...,None]*Cmul(Cmul(Cconj(Dk), Fy), Cconj(torch.rfft(grad_output, 2, onesided=False))))[...,None]), tuple(weights_shape[2:4])), dim=0)
        
        return grad_input, None, grad_lam, grad_weight
    
class WienerDeconvolution(nn.Module):
    def __init__(self, n_filters=8, filter_size=3, grayscale=False):
        super(WienerDeconvolution, self).__init__()
        
        self.lam = nn.Parameter(torch.rand(1).squeeze());
        
        self.function = WienerFunction;
        
        if grayscale:
            self.filters = nn.Parameter(nn.init.kaiming_normal_(torch.zeros(n_filters,1,filter_size,filter_size)));
        else:
            self.filters = nn.Parameter(nn.init.kaiming_normal_(torch.zeros(n_filters,3,filter_size,filter_size)));
        
    def forward(self,inputs):
        y, k = inputs;
        x = edgetaper_torch(pad_for_kernel_torch(y, k), k);
        
        return self.function.apply(x, k, self.lam, self.filters);