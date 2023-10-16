import os
import glob
import tabulate
from fpdf import FPDF
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter, label, find_objects
import skimage as sk
from skimage.io import imread, imshow
from skimage import exposure
from matplotlib.patches import Polygon

from fourier_scale_calibration import FourierSpaceCalibrator, windowed_fft

import tifffile

def make_scalebar(image, scale, subplots=(1,1,1)):
    real_length = round(scale*len(image)/5)
    real_length = round(real_length, -(len(str(real_length)) - 1))
    px_length = real_length/scale

    fig, axs = plt.subplots(subplots[0], subplots[1], figsize=(subplots[1]*5, subplots[0]*5))
    axs[subplots[2]-1].hlines(y=len(image)*30/31, xmin=px_length/5, xmax=px_length/5+px_length, lw=6, color='black')
    axs[subplots[2]-1].text(x=px_length/5, y=len(image)*29.5/31, s=f'{real_length} nm', color='black', fontsize=16, weight='bold');
    return fig, axs

class FourierAnalyzer:
    """
    A class that should take an image of some moire pattern (ideally close to 32x32 nm fov) and get the angle from the fft spots.
    
    Attributes:
        file (str) : the image file name
        image (np.array) : the image in question
        fourier (np.array) : shifted fourier transform of image
        scale (float) : the scale of the image
        angle (float) : the twist angle of the moire
        timestamp (datetime object) : the time and date of recording
        peaks (list): list of the diffraction peaks
        angles (list) : the angles between the spots
        vectors (list) : vectors corresponding to the peaks
        size : size of the chosen fov in nm

    Methods:
        get_timestamp : read timestamp from filename
        find_angles : a method that calculates the angle from the fft
        show_results : presents the image and angle in a nice way
    """
    
    def __init__(self, file, scale=None, size=16):
        self.file = file
        self.scale = scale #in nm per pixel
        
        self.image = imread(file) #read the image file
        self.fourier = np.fft.fftshift(np.abs(windowed_fft(self.image)))
        self.timestamp = self.get_timestamp() #get timestamp from filename

        self.peaks = None
        self.angles = []
        self.vectors = []
        
        self.size = size
        
        if self.size==16:
            if len(self.image)==2048:
                self.cutoff = 100
            elif len(self.image)==4096:
                self.cutoff = 200

        elif self.size==8:
            if len(self.image)==2048:
                self.cutoff = 50
            elif len(self.image)==4096:
                self.cutoff = 25
        
        if self.size==16:
            if len(self.image)==2048:
                self.parameters = {
                    'mask_radius' : (79, 84),
                    'neighborhood_size' : 6,
                    'threshold' : 3
                }
            elif len(self.image)==4096:
                self.parameters = {
                    'mask_radius' : (135, 158),
                    'neighborhood_size' : 7,
                    'threshold' : 1.3
                }
        elif self.size==8:
            if len(self.image)==2048:
                self.parameters = {
                    'mask_radius' : (36, 43),
                    'neighborhood_size' : 6,
                    'threshold' : 2.6
                }
            elif len(self.image)==4096:
                self.parameters = {
                    'mask_radius' : (16, 22),
                    'neighborhood_size' : 6,
                    'threshold' : 1.5
                }

    
    def get_timestamp(self):
        """ 
        get timestamp from filename 
        Returns:
            timestamp (datetime object) : the time and date of image recording
        """
        time = self.file.split('_')[-2].split('.')[0]
        time = datetime.strptime(time, '%Y-%m-%dT%H%M%S')
        return time
    
    def find_angles(self):
        
        mask_center = len(self.image)/2
        n = self.cutoff  #change how much is cut away from the plots of the fft
        
        #calculate the power spectrum for easier visualization and analysis#
        power_spec = abs(np.log(self.fourier))
        
        fourier = gaussian_filter(power_spec, sigma=.3) #slight filter
        
        
        #apply masks and filters#
        ix, iy   = np.meshgrid(np.arange(self.image.shape[0]), np.arange(self.image.shape[1]))
        distance = np.sqrt((ix - mask_center)**2 + (iy - mask_center)**2)

        img_mask = np.ma.masked_where(distance < self.parameters['mask_radius'][0], fourier) 
        img_mask = np.ma.masked_where(self.parameters['mask_radius'][1] < distance, img_mask) 

        ## finding the fft peaks ##

        data_max   = maximum_filter(img_mask, self.parameters['neighborhood_size'])
        maxima     = (img_mask == data_max)
        data_min   = minimum_filter(img_mask, self.parameters['neighborhood_size'])

        difference = ((data_max - data_min) > self.parameters['threshold']) 
        maxima[difference == 0] = False

        labeled, num_objects = label(maxima)
        slices = find_objects(labeled)

        peaklist = []
        for dy,dx in slices: # - assigns x- and y-coordinates to each maximum ''' 
            x_center = (dx.start + dx.stop - 1.) / 2.
            y_center = (dy.start + dy.stop - 1.) / 2.      
            peaklist.append([x_center, y_center])

        self.peaks = np.array(peaklist)
        
        
        ## find the angles between the vectors connecting the peaks ##
        vectors = []
        center = [len(self.image)/2, len(self.image)/2]

        for i, p in enumerate(self.peaks):
            vec = p-center
            u = vec/np.linalg.norm(vec)
            vectors.append((vec, u))

        for i, v in enumerate(vectors):
            for w in vectors[i+1:]:
                prod = np.dot(v[1], w[1])
                a = np.arccos(round(prod, 15))
                if (a > .5): #not very pretty but we just care about the small angles
                    continue
                self.angles.append(a)
                
        ## plot the whole thing ##
        if self.scale != None:
            fig, axs = make_scalebar(self.image, self.scale, (1,3,1))#plt.subplots(1,3, figsize=(15,5))
        
        axs[0].imshow(exposure.adjust_gamma(gaussian_filter(self.image, sigma=1), gamma=0.7), cmap='gray')
        axs[1].imshow(img_mask, cmap='gray')
        f = axs[2].imshow(power_spec, cmap='gray')
        plt.colorbar(f)
        axs[2].plot(self.peaks[:,0], self.peaks[:,1], 'x', color='red')
        for p in self.peaks:
            axs[2].plot([p[0], center[0]], [p[1], center[1]],'--', color='teal')
        
        for a in axs[1:]:
            a.set_xlim(len(self.fourier)/2-n, len(self.fourier)/2+n)
            a.set_ylim(len(self.fourier)/2-n, len(self.fourier)/2+n)



    def show_results(self,gaussian=0, alpha=None, log=None, equalize=False, save=False):
        """
        shows a comprehensive overview of the above calculated results
        """
        n = self.cutoff
        center = [len(self.image)/2, len(self.image)/2]
        
        table = f"\t{self.timestamp.strftime('%d-%m-%Y, %H:%M:%S')}\n"+\
                    f"angle\t\t\t:\t{round(np.degrees(np.mean(self.angles)), 1)}°"
        table = table.replace("\t", "    ")
        
        ## create nice image ##
        image = self.image
        
        if gaussian:
            image = sk.filters.gaussian(image, gaussian)
        if alpha:
            image = sk.exposure.adjust_gamma(image, alpha)
        if log:
            image = sk.exposure.adjust_log(image, log)
        if equalize:
            image = sk.exposure.equalize_hist(image)
        
        
        fig, axs = make_scalebar(image, self.scale, (1,2,1))
        
        axs[0].imshow(image, cmap='plasma')
        axs[0].axis('off')
        axs[1].imshow(abs(np.log(self.fourier)), cmap='gray')
        axs[1].plot(self.peaks[:,0], self.peaks[:,1], 'x', color='red')
        for p in self.peaks:
            axs[1].plot([p[0], center[0]], [p[1], center[1]],'--', color='teal')

        axs[1].set_xlim(len(self.fourier)/2-n, len(self.fourier)/2+n)
        axs[1].set_ylim(len(self.fourier)/2-n, len(self.fourier)/2+n)
        
        axs[1].axis('off')
        
        if save:
            name = self.timestamp.strftime('%Y%m%d_%H%M%S')+f'_moire_{round(np.degrees(np.mean(self.angles)))}.pdf'
            if type(save)==str:
                path = save + name
            else:
                path = os.path.split(self.file)[0] + '/'+ name
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
            print(path)
    
