import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy.constants import h, m_e, e, c


## a collection of useful helper functions ##

def make_scalebar(image, scale, subplots=(1,1,1), unit='nm'):
    if unit=='mrad':
        factor = 1e3
    else:
        factor = 1
    real_length = round(scale*len(image)*factor/5)
    real_length = round(real_length, -(len(str(real_length)) - 1))
    px_length = real_length/(scale*factor)

    fig, axs = plt.subplots(subplots[0], subplots[1], figsize=(subplots[1]*5, subplots[0]*5))
    if type(axs) == np.ndarray:
        axs = axs.ravel()
        if type(subplots[2]) == tuple:
            for n in subplots[2]:
                axs[n-1].hlines(y=len(image)*30/31, xmin=px_length/5, xmax=px_length/5+px_length, lw=6, color='black')
                axs[n-1].text(x=px_length/5, y=len(image)*29.5/31, s=f'{real_length} {unit}', color='black', fontsize=16, weight='bold');
                axs[n-1].axis('off')
        else:
            axs[subplots[2]-1].hlines(y=len(image)*30/31, xmin=px_length/5, xmax=px_length/5+px_length, lw=6, color='black')
            axs[subplots[2]-1].text(x=px_length/5, y=len(image)*29.5/31, s=f'{real_length} {unit}', color='black', fontsize=16, weight='bold');
            axs[subplots[2]-1].axis('off')
    else:
        axs.hlines(y=len(image)*30/31, xmin=px_length/5, xmax=px_length/5+px_length, lw=6, color='black')
        axs.text(x=px_length/5, y=len(image)*29.5/31, s=f'{real_length} {unit}', color='black', fontsize=16, weight='bold');
        axs.axis('off')

    return fig, axs
    
def hexagon(angle, a, center):
    """ 
    A regular hexagon with side length a twisted by an angle >angle< from the position
    of two spots lying on the x axis.
    """
    _p1 = np.array([0 + a*np.sin(angle), a*np.cos(angle)]) 
    _p2 = np.array([a*np.sin(np.pi/3 + angle), a*np.cos(np.pi/3 + angle)])
    _p3 = np.array([a*np.sin(np.pi*2/3 + angle), a*np.cos(np.pi*2/3 + angle)])
    _p4 = -_p1
    _p5 = -_p2
    _p6 = -_p3

    #_p = [_p5, _p2, _p6, _p3, _p4, _p1]
    _p = [_p1, _p2, _p3, _p4, _p5, _p6]
    _p = [p + np.array(center) for p in _p]
    return _p

def cart2pol(x, y, center):
    """
    Takes cartesian coordinates given by x, y and returns
    the radius and angle phi of the point in polar coordinates
    with origin center.
    Args:
        x (int) : x coordinate of image point
        y (int) : y coordinate of image point
        center (tuple) : (x, y) coordinates of center for polar coordinates
    Returns:
        (r, phi) : tupel of the radius and angle in radians
    """
    x = x-center[0]
    y = y-center[1]
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(r, phi)

def pol2cart(r, phi, center):
    """
    Takes polar coordinates r and phi around a center and returns
    the cartesian coordinates of the given point with origin lower
    left corner.
    Args:
        r (int) : radius of the point in polar coordinates
        phi (float) : angle in radians of the point in p.c.
        center (tupel) : center in cartesian coordinates around
                            which the p.c. are oriented
    Returns:
        (x, y) : tupel of x and y cartesian coordinates
    """
    x = r * np.cos(phi) + center[0]
    y = r * np.sin(phi) + center[1]
    return (x, y)

def mask(im, r_in, r_out=None, center=None):
    """
    masks a given image around an optionally given center 
    Args:
        im (np.array) : array corresponding to the image to be masked
        r_in (int) : number corresponding to the inner radius of the mask in px
        r_out (int) : number corresponding to outer radius of mask (default None -
                        mask is disk not annulus)
        center (tupel) : tupel corresponding to the mask center (default None - use
                        use the center of the image)
    Returns:
        img_mask (np.array) : the array corresponding to the masked image
    """
    if center==None: # if no center is given
        mask_center = (im.shape[0]//2, im.shape[1]//2)

    ix, iy   = np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]))
    distance = np.sqrt((ix - mask_center[0])**2 + (iy - mask_center[1])**2)

    img_mask = np.where(distance < r_in, 0, im) 
    
    if r_out: # if mask is an annulus
        img_mask = np.where(distance > r_out, 0, img_mask)
    
    return img_mask

def threshold(im, thresh, gaussian=3, mask_radius=100, verbose=False, scale=False, save=False):
    """
    thresholds an image with a tuple of lower and higher thresholds
    and an optionally given gaussian filter with sigma gaussian
    Args:
        im (np.array) : array corresponding to the image
        thres (tupel) : a tuple with a lower and higher threshold value
        gaussian (int/bool) : if False no filter is applied, if number, gaussian
                                with sigma of the number is applied (default : 3)
        mask_radius (int/bool) : if False no mask is applied, otherwise the image is 
                                masked with an inner radius of mask_radius (default : 100)
        verbose (bool) : if True more information is shown
    Returns:
        binary (np.array) : a binary image corresponding to the thresholded original
    """

    if bool(gaussian)==True:
        im = skimage.filters.gaussian(im, sigma=gaussian)
    
    if bool(mask_radius)==True:
        im = mask(im, mask_radius)
    
    
    thresh_l = thresh[0]
    thresh_h = thresh[1]
    binary = (im > thresh_l) & (im < thresh_h)
    
    if verbose==True:
        if scale:
            fig, axes = make_scalebar(im, scale, subplots=(1,3,(1,3)), unit='mrad') 
            
        else:
            fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
            
        ax = axes.ravel()
        ax[0] = plt.subplot(1, 3, 1)
        ax[1] = plt.subplot(1, 3, 2)
        ax[2] = plt.subplot(1, 3, 3)

        ax[0].imshow(im.transpose(), cmap='plasma')
        ax[0].set_title('Original')
        ax[0].axis('off')

        ax[1].hist(im.ravel(), bins=256, density=True)
        ax[1].set_title('Histogram')
        ax[1].axvline(thresh_l, color='r')
        ax[1].axvline(thresh_h, color='r')

        ax[2].imshow(binary.transpose(), cmap='plasma')
        ax[2].set_title('Thresholded')
        ax[2].axis('off')

        #plt.tight_layout
        
    if save:
        plt.savefig(save, bbox_inches='tight', pad_inches=0)
        
    
    return binary

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2, small=True):
    """ Returns the angle in radians between vectors 'v1' and 'v2'
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    a = [np.arccos(np.dot(v1_u, v2_u)), np.arccos(np.dot(-v1_u, v2_u))]

    if small==True:
        return min(a)
    else:
        return a[0]

def bragg_angle(d, energy):
    """ Calculates the Bragg angle for electron energy in keV and distance d in angstrom """

    _d = d*1e-10 #convertion to m
    _energy = energy*e*1e3 #coversion to Joule
    _lambda_el = h/np.sqrt( 2*m_e*_energy * (1 + _energy/(2*m_e*c**2))) #relativistic electron wavelength

    theta = np.arcsin(_lambda_el/(2*_d))
    return theta
