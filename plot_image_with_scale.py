import matplotlib.pyplot as plt
import skimage
import pandas as pd
import tifffile
from helper_functions import make_scalebar

def get_scale(file, scale_df):
    with tifffile.TiffFile(file) as tif:
        tif_tags = {}
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            if name=='ImageDescription':
                tif_tags[name] = value
    image = tif.pages[0].asarray()
    pre_scale=float(tif_tags['ImageDescription'].split(',')[1].split(':')[-1])
    pre_scale = int(len(image)*pre_scale)

    scale = scale_df.loc[ f'{int(pre_scale)}x{int(pre_scale)}',str(len(image))]
    return scale

def plot(file, scale_df, gaussian=0, alpha=None, log=None,equalize=False, save=False):
    scale = get_scale(file, scale_df)
    image = skimage.io.imread(file)
    
    if gaussian:
        image = skimage.filters.gaussian(image, gaussian)
    if alpha:
        image = skimage.exposure.adjust_gamma(image, alpha)
    if log:
        image = skimage.exposure.adjust_log(image, log)
    if equalize:
        image = skimage.exposure.equalize_hist(image)
    
    fig, axs = make_scalebar(image, scale, (1,1,1), unit='nm')
    plt.axis('off')
    axs.imshow(image, cmap='plasma')
    
    if save:
        plt.savefig(save, bbox_inches='tight', pad_inches=0)
        print(save)
        
    
    fig.show()
