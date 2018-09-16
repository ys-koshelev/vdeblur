import torch

# Initialization of a PyTorch complex tensor
'''
TWO TIMES SLOWER VERSION IF EVALUATING ON CPU
def Cinit(real=None,imag=None):
    """
    Converts two real PyTorch tensors to a complex one by unsqueezing and then concatenating in last dimension
    
    Parameters
    ----------
    real[torch.(cuda)Tensor]: PyTorch tensor of an arbitrary shape and type, default=None 
        Tensor containing a real part
    imag[torch.(cuda)Tensor]: PyTorch tensor of an arbitrary shape and type, default=None 
            Tensor containing an imaginary part
            
    Returns
    -------
    [torch.(cuda).Tensor]: None if both inputs are None, double sized tensor with last dimension of size 2 with real and imaginary parts
        Corresponding complex tensor
    """
    if (real is None)&(imag is None):
        return None;
    elif imag is None:
        imag = real.new(real.shape).fill_(0);
    elif real is None:
        real = imag.new(imag.shape).fill(0);
        
    return torch.cat((real.unsqueeze(-1),imag.unsqueeze(-1)),dim=-1);
'''

def Cinit(real=None,imag=None):
    """
    Converts two real PyTorch tensors to a complex one by unsqueezing and then concatenating in last dimension
    
    Parameters
    ----------
    real[torch.(cuda)Tensor]: PyTorch tensor of an arbitrary shape and type, default=None 
        Tensor containing a real part
    imag[torch.(cuda)Tensor]: PyTorch tensor of an arbitrary shape and type, default=None 
            Tensor containing an imaginary part
            
    Returns
    -------
    ret[torch.(cuda).Tensor]: None if both inputs are None, double sized tensor with last dimension of size 2 with real and imaginary parts
        Corresponding complex tensor
    """
    if (real is None)&(imag is None):
        return None;
    
    ret = real.new(real.shape + (2,)).fill_(0);
    if imag is not(None):
        ret[...,1] = imag;
    if real is not(None):
        ret[...,0] = real;
    return ret;

# Operations, converting PyTorch complex tensor to several real ones
def Creal(input):
    """
    Takes real part of PyTorch complex tensor
    
    Parameters
    ----------
    input[torch.(cuda)Tensor]: PyTorch complex tensor of an arbitrary type and shape with 2 channels in last dimension
            
    Returns
    -------
    [torch.(cuda).Tensor]: PyTorch tensor of the same shape as an input, but without the last dimension,
        Real part of corresponding PyTorch complex tensor
    """
    assert input.shape[-1] == 2, 'Invalid input: not a complex number, complex tensor is assumed to have 2 channels in last dimension!'
    
    return input[...,0];

def Cimag(input):
    """
    Takes imaginary part of PyTorch complex tensor
    
    Parameters
    ----------
    input[torch.(cuda)Tensor]: PyTorch complex tensor of an arbitrary type and shape with 2 channels in last dimension
            
    Returns
    -------
    [torch.(cuda).Tensor]: PyTorch tensor of the same shape as an input, but without the last dimension,
        Imaginary part of corresponding PyTorch complex tensor
    """
    assert input.shape[-1] == 2, 'Invalid input: not a complex number, complex tensor is assumed to have 2 channels in last dimension!'
    
    return input[...,1];

def Cabs(input):
    """
    Calculates absolute value of PyTorch complex tensor
    
    Parameters
    ----------
    input[torch.(cuda)Tensor]: PyTorch complex tensor of an arbitrary type and shape with 2 channels in last dimension
            
    Returns
    -------
    [torch.(cuda).Tensor]: PyTorch tensor of the same shape as an input, but without the last dimension,
        Absolute value of corresponding PyTorch complex tensor
    """
    assert input.shape[-1] == 2, 'Invalid input: not a complex number, complex tensor is assumed to have 2 channels in last dimension!'
    
    return torch.norm(input,2,-1);

# Pixelwise arithmetic operations under PyTorch complex tensors, which cannot be expressed by standard pixelwise operations
def Cmul(x,y):
    """
    Calculates a complex multiplication two PyTorch complex tensor
    
    Parameters
    ----------
    x,y[torch.(cuda)Tensor]: PyTorch complex tensors of an arbitrary type and shape with 2 channels in last dimension
            
    Returns
    -------
    ret[torch.(cuda).Tensor]: PyTorch complex tensor of the same type and shape as inputs,
        Value of x*y
    """
    
    ret = x.new(x.shape);
    ret[...,0] = Creal(x)*Creal(y) - Cimag(x)*Cimag(y);
    ret[...,1] = Creal(x)*Cimag(y) + Cimag(x)*Creal(y);
    return ret;

def Cdiv(numerator,denominator):
    """
    Calculates a complex division of two PyTorch complex tensor
    
    Parameters
    ----------
    numerator,denominator[torch.(cuda)Tensor]: PyTorch complex tensors of an arbitrary type and shape with 2 channels in last dimension
            
    Returns
    -------
    ret[torch.(cuda).Tensor]: PyTorch complex tensor of the same type and shape as inputs,
        Value of numerator/denominator
    """
    
    ret = numerator.new(numerator.shape);
    norm_denom = Creal(denominator)**2 + Cimag(denominator)**2;
    ret[...,0] = (Creal(numerator)*Creal(denominator) + Cimag(numerator)*Cimag(denominator))/(norm_denom);
    ret[...,1] = Cimag(numerator)*Creal(denominator) - Creal(numerator)*Cimag(denominator)/(norm_denom);
    return ret;

def Cconj(input):
    """
    Calculates a complex conjugation of PyTorch complex tensor
    
    Parameters
    ----------
    input[torch.(cuda)Tensor]: PyTorch complex tensor of an arbitrary type and shape with 2 channels in last dimension
            
    Returns
    -------
    ret[torch.(cuda).Tensor]: PyTorch complex tensor of the same type and shape as input,
        Value of input*, where * means complex conjugation
    """
    assert input.shape[-1] == 2, 'Invalid input: not a complex number, complex tensor is assumed to have 2 channels in last dimension!'
    
    ret = input.new(input.shape);
    ret[...,0] = Creal(input);
    ret[...,1] = -Cimag(input);
    return ret;