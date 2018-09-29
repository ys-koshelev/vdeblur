import torch
from io import BytesIO
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision.transforms import ToTensor
from PIL.Image import open as imopen

def colorbar(mappable):
    ax = mappable.axes;
    fig = ax.figure;
    divider = make_axes_locatable(ax);
    cax = divider.append_axes("right", size="5%", pad=0.05);
    return fig.colorbar(mappable, cax=cax);

def plot2writer(fig):
    """Export pyplot figure to tensor image for writing in TensorBoardX."""
    buf = BytesIO();
    fig.savefig(buf, format='png');
    buf.seek(0);
    image = imopen(buf);
    image = ToTensor()(image);
    plt.close('all');
    return image;

def visualise_weights(layer_weights,nv=6,nh=4,w=6.75,h=10):
    fig, axes = plt.subplots(nv, nh, figsize=(w, h));
    for i in range(nv):
        for j in range(nh):
            im = axes[i][j].imshow(layer_weights[i*nh+j,0,:,:]);
            axes[i][j].axis('off');
            colorbar(im);
    return plot2writer(fig);