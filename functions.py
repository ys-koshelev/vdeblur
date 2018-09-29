import torch
import torch.nn as nn
from Ctorch import Cinit, Cabs

def psf2otf_torch(psf, out_shape):
    """
    Convert point-spread function to optical transfer function.

    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.

    To ensure that the OTF is not altered due to PSF off-centering, psf2otf
    post-pads the PSF array with zeros in the way to disperse both halves of 
    the signal to different borders in both vertical and horizontal directions.
    
    Parameters
    ----------
    psf[torch.(cuda).Tensor]: batch of real-valued tensor of shape [out_channels,in_channels,H,W]
        Point Spread Function
    out_shape[tuple or torch.Size]: 2 numbers in the order of H,W
        Shape of the output Optical Transfer Function

    Returns
    -------
    otf[torch.(cuda).Tensor]: complex-valued tensor of shape [out_channels,in_channels,out_shape[0],out_shape[1],2]
        Optical Transfer Function
    """
    
    # Shape of the point spread function(s)
    psf_shape = psf.shape;

    # Coordinates for splitting the psf tensor
    midH = psf_shape[2]//2;
    modH = psf_shape[2]%2;
    midW = psf_shape[3]//2;
    modW = psf_shape[3]%2;
    
    # Extending and filling PSF assuming periodic boundaries
    pre_otf = psf.new(psf.shape[:2] + out_shape).fill_(0);
    pre_otf[:,:,-midH:, -midW:] = psf[:,:,:midH, :midW];
    pre_otf[:,:,-midH:, :(midW + modW)] = psf[:,:,:midH, midW:];
    pre_otf[:,:,:(midH + modH), -midW:]  = psf[:,:,midH:, :midW];
    pre_otf[:,:,:(midH + modH), :(midW + modW)] = psf[:,:,midH:, midW:];
    
    # Fast fourier transform, transposed because tensor must have shape [..., height, width] for this
    otf = torch.rfft(pre_otf,signal_ndim=2, onesided=False);

    return otf

def otf2psf_torch(otf,out_shape):
    """
    Convert optical transfer function to point-spread function.
    
    Computes the inverse Fast Fourier Transform (IFFT)
    of the optical transfer function (OTF) array and creates a point spread
    function (PSF), centered at the origin.

    To center the PSF at the origin, otf2psf circularly shifts the values
    of the output array down (or to the right) until the (1,1) element
    reaches the central position, then it crops the result to match
    dimensions specified by out_shape.

    Parameters
    ----------
    otf[torch.(cuda).Tensor]: batch of complex-valued tensor of shape [out_channels,in_channels,H,W,2]
        Optical Transfer Function
    out_shape[tuple or torch.Size]: 2 numbers in the order of H,W
        Shape of the output Point Spread Function

    Returns
    -------
    otf[torch.(cuda).Tensor]: complex-valued tensor of shape [out_channels,in_channels,out_shape[0],out_shape[1]]
        Point Spread Function
    """
    # reserving memory for PSF tensors
    psf = otf.new(otf.shape[:-3] + out_shape);
    
    # Coordinates with which the PSF tensors were splitted
    midH = out_shape[0]//2;
    modH = out_shape[0]%2;
    midW = out_shape[1]//2;
    modW = out_shape[1]%2;
    
    # Converting OTFs to a real space
    pre_psf = torch.irfft(otf, 2, onesided=False);
    
    # Extending and filling PSFs assuming periodic boundaries
    psf[:,:,:midH, :midW] = pre_psf[:, :,-midH:, -midW:];
    psf[:,:,:midH, midW:] = pre_psf[:, :,-midH:, :(midW + modW)];
    psf[:,:,midH:, :midW] = pre_psf[:, :, :(midH + modH), -midW:];
    psf[:,:,midH:, midW:] = pre_psf[:, :, :(midH + modH), :(midW + modW)];
    
    return psf;

def pad_for_kernel_torch(imgs,kernels):
    shape = kernels.shape;
    p_h = (shape[-2]-1)//2;
    p_w = (shape[-1]-1)//2;
    return nn.ReplicationPad2d((p_h,p_h,p_w,p_w))(imgs);

def pad_for_kernel_circ_torch(imgs,kernels):
    shape = kernels.shape;
    p_h = (shape[-2]-1)//2;
    p_w = (shape[-1]-1)//2;
    
    i_shape = list(imgs.shape);
    i_shape[2] = i_shape[2] + 2*p_h;
    i_shape[3] = i_shape[3] + 2*p_w;
    ret = imgs.new(*i_shape);
    
    ret[:,:,p_h:-p_h,p_w:-p_w] = imgs[:,:,:,:];
    
    ret[:,:,:p_h,p_w:-p_w] = imgs[:,:,-p_h:,:];
    ret[:,:,-p_h:,p_w:-p_w] = imgs[:,:,:p_h,:];
    
    ret[:,:,p_h:-p_h,:p_w] = imgs[:,:,:,-p_w:];
    ret[:,:,p_h:-p_h,-p_w:] = imgs[:,:,:,:p_w];
    
    ret[:,:,:p_h,:p_w] = imgs[:,:,-p_h:,-p_w:];
    ret[:,:,:p_h,-p_w:] = imgs[:,:,-p_h:,:p_w];
    ret[:,:,-p_h:,:p_w] = imgs[:,:,:p_h,-p_w:];
    ret[:,:,-p_h:,-p_w:] = imgs[:,:,:p_h,:p_w];
    return ret;

'''
def pad_for_kernel_circ_torch(imgs,kernels):
    shape = kernels.shape;
    i_shape = imgs.shape;
    
    p_h = (shape[-2]-1)//2;
    p_w = (shape[-1]-1)//2;
    
    return imgs.repeat(1,1,3,3)[:,:,(i_shape[-2]-1-p_h):(2*i_shape[-2]-1+p_h),(i_shape[-1]-1-p_w):(2*i_shape[-1]-1+p_w)];
'''

def crop_for_kernel_torch(imgs,kernels):
    k_shape = kernels.shape;
    p_h = (k_shape[-2]-1)//2;
    p_w = (k_shape[-1]-1)//2;
    return imgs[:,:,slice(p_h,-p_h),slice(p_w,-p_w)];

def edgetaper_alpha_torch(kernels,img_shape):
    v = [];
    for i in range(2):
        s = torch.sum(kernels,dim=3-i);
        z = torch.rfft(nn.ConstantPad1d((0,img_shape[i]-1-s.shape[-1]),0)(s),1,onesided=False);
        z = torch.irfft(Cinit(real=Cabs(z)**2),1,onesided=False);
        z = torch.cat((z,z[:,:,0:1]),2).squeeze(1);
        v.append(1 - z/torch.max(z));
    return torch.bmm(v[0].unsqueeze(2), v[1].unsqueeze(1)).unsqueeze(1);

def flip_kernels(kernels):
    return torch.flip(kernels,(3,2));

def edgetaper_torch(imgs,kernels,n_tapers=3):
    alphas = edgetaper_alpha_torch(kernels, imgs.shape[-2:]);
    if imgs.shape[1] == 3:
        a_shape_new = list(alphas.shape);
        a_shape_new[1] == 3;
        alphas  = alphas.expand(a_shape_new);
    for i in range(n_tapers):
        blurred = nn.functional.conv2d(pad_for_kernel_circ_torch(imgs,kernels).permute(1,0,2,3),
                                       flip_kernels(kernels),groups=kernels.shape[0]).permute(1,0,2,3);
        imgs = alphas*imgs + (1-alphas)*blurred;
    return imgs;

def adjust_bounds(x,y,k):
    k_shape = k.shape;
    p_h = ((k_shape[-2]-1)//2);
    p_w = ((k_shape[-1]-1)//2);
    x_blurred = nn.functional.conv2d(pad_for_kernel_circ_torch(x,k).permute(1,0,2,3),
                                       flip_kernels(k),groups=k.shape[0]).permute(1,0,2,3);
    
    x_blurred[:,:,p_h:-p_h,p_w:-p_w] = y[:,:,:,:];
    return x_blurred;