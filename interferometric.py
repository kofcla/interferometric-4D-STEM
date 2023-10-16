### This is a collection of analysis code for 4D STEM interferometric analysis ###

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm

import skimage
from skimage.io import imread, imsave
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square

import scipy
import scipy.optimize as opt
from scipy.constants import h, m_e, e, c
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter, find_objects

import json

import os

from fourier_scale_calibration import FourierSpaceCalibrator, windowed_fft

import dask
from dask import array

import h5py

import py4DSTEM

from tqdm import tqdm

import helper_functions as hf


def save_DataAnalyzer_object(savepath, DA_object):
    """
    A method to save an existing DataAnalyzer class object and connected data items.
    Args:
        savepath (str) : string describing the path to the directory where the object
                            should be saved
        DA_object (object) : object of the DataAnalyzer instance
    """
    
    while True:
        try:
            os.mkdir(savepath) #create a new directory at the path
            file_path = os.path.join(savepath, 'datacube.h5')
            with h5py.File(file_path, 'w') as h5_file:
                h5_data = h5_file.create_group('data')
                h5_data.create_dataset('dp_mean', data=DA_object.dp_mean.data)

                h5_info = h5_file.create_dataset('info', (1,))
                h5_info.attrs['scale'] = np.asarray(DA_object.scale)
                h5_info.attrs['energy'] = np.asarray(DA_object.energy)
                h5_info.attrs['twist_angle'] = np.asarray(DA_object.twist_angle)
                h5_info.attrs['center'] = DA_object.center
                h5_info.attrs['disk_r'] = DA_object.disk_r
                h5_info.attrs['disk_mask'] = DA_object.disk_mask
                h5_info.attrs['r_outer'] = DA_object.r_outer
                h5_info.attrs['r_inner'] = DA_object.r_inner
                h5_info.attrs['path'] = DA_object.path

                h5_images = h5_file.create_group('images')
                h5_images.create_dataset('real_image', data=DA_object.real_image)
                h5_images.create_dataset('virtual_image', data=DA_object.virtual_image.data) 
                h5_images.create_dataset('label_image', data=DA_object.label_image)
            break

        except FileExistsError: #if the file exists 
            overwrite = input("This directory already exist, do you want to overwrite it? (y/n) ")
            if overwrite=='n':
                break
            elif overwrite=='y':
                files = os.listdir(savepath)
                for file in files:
                    file_path = os.path.join(savepath, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(savepath)
                continue

def load_DataAnalyzer_object(savepath):
    """
    Loads a saved DataAnalyzer object from some savepath.

    Args:
        savepath (str) : path to the save directory
    
    Returns:
        object (object) : DataAnalyzer object constructed from the saved data.
    """

    file_path = os.path.join(savepath, 'datacube.h5')
    with h5py.File(file_path, 'r') as h5_file:
        dp_mean = h5_file.get('data/dp_mean')[:]
        info = h5_file['info']
        info_dict = dict(list(info.attrs.items()))
        real_image = h5_file.get('images/real_image')[:]
        virtual_image = h5_file.get('images/virtual_image')[:]
        label_image = h5_file.get('images/label_image')[:]

    object = DataAnalyzer(info_dict['path'], info_dict['energy'])
    object.dp_mean = dp_mean
    object.real_image = real_image
    object.virtual_image = virtual_image
    object.label_image = label_image
    
    for k in info_dict.keys():
        setattr(object,k,info_dict[k])
        
    return object

        
class DataAnalyzer:
    
    """
    A class to analyze interferometric 4D STEM data. 
    
    Attributes:
    ----------------------
    path (str)                  :   the path to the data to be analyzed
    energy (float)              :   energy in keV of the experiment
    verbose (bool)              :   if it is True, more data will be shown
    disk_mask (float)           :   the mask that is applied to the disks 
                                    before fitting in mrad
    datacube (py4DSTEM object)  :   the data to be analyzed
    metadata (json dict)        :   a json dictionary containing the 
                                    experimental metadata
    dp_mean (array)             :   the mean diffraction pattern
    center (tuple)              :   the center of the diffraction pattern
    disk_r (float)              :   the radius of the disks in px corresponding 
                                    to conv. angle
    dp_shape (tuple)            :   the shape of the diffraction patterns
    cube_shape (tuple)          :   the shape of the recorded map
    pairs (list)                :   a list of the diffraction disk pairs
    label_image (array)         :   the calculated lable image mask
    r_outer (float)             :   the radius of the outer graphene disks in px
    r_inner (float)             :   the radius of the inner graphene disks in px
    scale (float)               :   the scale of the image 
                                    (px * scale = distance in rad) 
    angles (list)               :   list of the angles of the different spots
    real_image (array)          :   MAADF comparison image
    image_scale (float)         :   scale of the MAADF comparison image
    twist_angle (float)         :   twist angle of the structure in rad
    virtual_image (array)       :   reconstructed virtual ADF image
    binned_data (array)         :   data binned by a factor of 2
    virtual_image_binned (array):   image reconstructed from binned data

    Methods:
    ----------------------
    calculate_mean              :   a method to calculate the average diffraction pattern
    load_file                   :   a method to load data and metadata from an h5 file
    calculate_data              :   calculates a number of properties from the dataset
    get_convergence_angle       :   finds the convergence angle in px from the mean pattern   
    create_binary_image_from_stack: takes a datacube, thresholds a given number of images, sums
                                    them and outputs a image where the disks should be decernable
    get_label_image             :   takes the binary image, fits a hexagon to find the centers of
                                    the diffraction disks and calculates the scale
    calculate scale             :   calculates the scale of the diffraction patterns from 
                                    the Bragg angle 
    bin_data_new                :   binns the data by a factor of 2 using for loops
    bin_data                    :   binns the data by reshaping the arrays and averaging then
    reconstruct_adf             :   reconstructs an adf image with given inner and outer radii 
    show_images                 :   shows and saves the virtual and real image
    visualize_all               :   plot and save images of all important steps
    fit_position                :   fits cosine function to a specific position of the datacube
    analyze_map                 :   analyzes the whole datacube by fitting every position
    get_fitted_maps             :   uses the fit parameters to return arrays of all fit parameters
                                    and the calculated ild and defocus
    analyze_disks               :   returns fit parameters and calculated values for all given disks
    """
    
    def __init__(self, path, energy, image_path=None, image_scale=None, verbose=False):
        self.path = path #the path to the data to be analyzed
        self.energy = energy #electron energy in keV
        self.verbose = verbose
        
        self.disk_mask = 8

        self.datacube, self.metadata = self.load_file() #the complete data cube 
        self.dp_mean = None #the mean diffraction pattern
        
        self.center = None #the center of the image
        self.disk_r = None #the radius of the disk, corresponding to conv. angle
        self.dp_shape = self.datacube.shape[2:] #the shape of the dp
        self.cube_shape = self.datacube.shape[:2] #the shape of the cube
        
        self.pairs = None
        
        self.label_image = None #the calculated lable image mask
        self.r_outer = None #the radius of the outer graphene disks
        self.r_inner = None #the radius of the inner graphene disks
        self.scale = None #the scale of the image (px * scale = distance in rad) 
        self.angles = None #list of the angles of the different spots
        
        if image_path !=None:
            extension = image_path.split('.')[-1]
            if extension=='npy':
                self.real_image = np.load(image_path)
            elif extension=='hdf5':
                with h5py.File(image_path, 'r') as f:
                    im = f.get('array')[:]
                    sampling = f.get('sampling')[:]
                self.real_image = skimage.transform.rescale(im, tuple(sampling))
            elif extension=='tif':
                self.real_image = imread(image_path)
        self.image_scale = image_scale
            
            
        self.twist_angle = None
        
        self.virtual_image = None
        
        self.binned_data = None

        self.virtual_image_binned = None
        
    def calculate_mean(self):

        self.dp_mean = self.datacube.get_dp_mean().data.compute()
        
    def load_file(self):
        # has to be changed for other file formats
        x = h5py.File(self.path)
        try:
            data = array.from_array(x["data"])
            datacube = py4DSTEM.DataCube(data)
            metadata = json.loads(x['data'].attrs['properties'])
        except KeyError:
            datacube = array.from_array(x["array"])
            
            datacube = array.from_array(skimage.transform.rescale(datacube, 
                                    (1,1,x['sampling'][2]/x['sampling'][-1], 1)))
            datacube = py4DSTEM.DataCube(datacube)
            metadata = None
            
        return datacube, metadata
    
    def get_convergence_angle(self, threshold=(2e4,1e5), gaussian=5):

        ## threshold the mean diffraction pattern to find the center spot ##
        thresholded = hf.threshold(self.dp_mean, threshold, gaussian=gaussian, 
                                mask_radius=False, verbose=self.verbose, scale=self.scale)
            
        label_im = label(thresholded)
        
        self._conv_params = (threshold, gaussian)
        
        if self.verbose==True:
            plt.imshow(label_im.transpose(), cmap='plasma')
            
        regions = skimage.measure.regionprops(label_im, self.dp_mean)
        
        ## the radius corresponding to the convergence angle ##
        r = regions[0]['equivalent_diameter_area']/2 #equivalent radius of circle

        center = regions[0]['centroid']
        
        return r, center
        
    def create_binary_image_from_stack(self, line=0, first_thresholds=(15,60), 
                second_thresholds=(20,60), gaussian=3, mask_radius=100, save=False):
        
        thresholded = []
        # threshold a number of patterns until the line > line < #
        patterns = self.datacube[:,:line].reshape((self.cube_shape[0]*line, *self.dp_shape))
        for i in tqdm(range(len(patterns))):
            im = patterns[i] - self.dp_mean
            thresholded.append(hf.threshold(im, 
                                            first_thresholds, 
                                            gaussian=gaussian, 
                                            mask_radius=mask_radius)) 
        # sum all images thresholded this way #
        summed = np.sum(thresholded, axis=0)
        
        # do anothr threshold to get a binary image again #
        bin_image = hf.threshold(summed, 
                                 second_thresholds, 
                                 gaussian=None, 
                                 mask_radius=mask_radius, 
                                 verbose=self.verbose) 
                                       
        if self.verbose==True:
            if self.scale:
                fig, axs = hf.make_scalebar(summed, self.scale, (1,3,(1,3)), unit='mrad')
            else:
                fig, axs = plt.subplots(1,3, figsize=(8,3))
                
            axs[1].hist(summed.ravel(), bins=256);
            axs[0].imshow(summed.transpose(), cmap='plasma')
            axs[2].imshow(bin_image.transpose(), cmap='gray')
            plt.tight_layout()
            
            
        self._bin_image_steps = (summed, bin_image)

        return summed, bin_image

    def get_label_image(self, bin_image, square_px=5, min_area=2000):
        
        ## clean up the binary image calculated before ##
        bw = closing(bin_image, square(square_px))
        template = label(bw)

        ## measure the regions corresponding to the disks ##
        regions = skimage.measure.regionprops(template, bin_image)

        # filter out small areas #
        regions = [r for r in regions if r['area']>min_area] 

        # get the centers of all disks #
        centroids = np.array([list(r['centroid']) for r in regions])

        # get the mean distance of the points to the center #
        radii_all = [hf.cart2pol(c[0], c[1], self.center)[0] for c in centroids]
        r = np.mean(radii_all)

        # seperate inner and outer disks #
        outer = [np.array(c) for c in centroids if hf.cart2pol(*c, self.center)[0] > r ]
        inner = [np.array(c) for c in centroids if hf.cart2pol(*c, self.center)[0] < r ]


        ## mean radii estimation ##
        r_outer = np.mean([hf.cart2pol(c[0], c[1], self.center)[0] for c in outer])
        r_inner = np.mean([hf.cart2pol(c[0], c[1], self.center)[0] for c in inner])

        # create a first estimate #
        p = np.array(hf.hexagon(-np.radians(10), r_outer, self.center))
        
        # sort the points so that they agree with the order #
        sorted_outer = []
        for point in p:
            distances = [np.linalg.norm(point-o) for o in outer]
            idx = np.argmin(distances)
            sorted_outer.append(outer[idx])
            outer.pop(idx)
        
        outer = sorted_outer

        def minimize_function(params, point_estimate):
            comparison = hf.hexagon(*params[:2], params[2:])
            distances = [np.linalg.norm(comparison[i]-point_estimate[i]) for i in range(6)]
            return sum(distances)

        # give first estimates #
        p0 = [-np.radians(10), r_outer, self.center[0], self.center[1]]
        
        # fit the hexagon to the points #
        result = opt.minimize(minimize_function, p0, args=(outer))
        orientation, r_outer = result.x[:2]
        center = result.x[2:]
        outer = np.array(hf.hexagon(orientation, r_outer, center))

        if self.verbose == True:
            fig, axs = plt.subplots()
            axs.imshow(bin_image.transpose(), cmap='plasma')
            circle = plt.Circle(center, r_outer, edgecolor='red', facecolor=None, 
                                fill=False, ls='--')
            axs.plot(outer[:,0], outer[:,1], 'x')
            #axs.plot(outer[:,1], outer[:,0], 'x')
            axs.add_patch(circle)
        
        self.orientation = orientation
        self.r_outer = r_outer
        self.r_inner = r_inner
        self.center = center
        
        self._outer_spots = outer
        
        self._all_regions = regions
        
        ## calculate scale ##
        self.scale = self.calculate_scale() 
    
        ## create an image with labeled regions corresponding to disks ##
        label_image = np.zeros(self.dp_shape, dtype=int) #initialize empty image
        
        # disk radius in px #
        r_disk = self.disk_mask*1e-3/self.scale
        for i, c in enumerate(outer):#disk_coordinates):
            rr, cc = skimage.draw.disk((c[0], c[1]), r_disk)
            label_image[rr, cc] = int(i+1)

        if self.verbose==True:
            fig, axs = plt.subplots(1,2)
            axs[0].imshow(template.transpose(), cmap='plasma')
            axs[0].plot(outer[:,0], outer[:,1], 'x')
            axs[1].imshow(label_image.transpose(), cmap='plasma')
            axs[1].plot(outer[:,0], outer[:,1], 'x') 
                                       
        self.label_image = label_image
        
        ## save the symmetric disks ##
        regions = skimage.measure.regionprops(label_image)
        self.pairs = [
            (regions[0], regions[3]),
            (regions[1], regions[4]),
            (regions[2], regions[5])
        ]
        
    def calculate_scale(self):
        """
        Method to calculate the scale of the images using the radii of the graphene
        disks and the equaiton for the bragg angle.
        """
        
        a = 2.46 #in angstrom [find citation]
        
        ## find the right scale of the image ##
        
        # calculate the plane scales for zig-zag and armchair #
        d_zz = a*np.sin(np.pi/3)
        d_ac = a/2
        
        # calculate the bragg angle for those distances #
        inner_angle = hf.bragg_angle(d_zz, self.energy) #bragg angle for inner spots at 60 keV
        outer_angle = hf.bragg_angle(d_ac, self.energy) #outer spots

        scale_1 = 2*inner_angle/self.r_inner
        scale_2 = 2*outer_angle/self.r_outer
        
        return np.mean([scale_1, scale_2])
    
    def bin_data_new(self):
        empty = dask.array.empty((self.cube_shape[0]-1, self.cube_shape[1]-1, self.dp_shape[0], self.dp_shape[1]))
        for i in tqdm(range(self.cube_shape[0]-1)):
            for j in range(self.cube_shape[1]-1):
                mean_pattern = np.mean([self.datacube[i,j], self.datacube[i,j+1], self.datacube[i+1,j], self.datacube[i+1,j+1]],
                        axis=0)
                empty[i,j] = mean_pattern

        self.binned_data = empty
        self.binned_computed = empty.compute()

    def bin_data(self, bin_size=(2,2)):
        data = self.datacube.data

        new = array.random.random((data.shape[0]-1, data.shape[1]-1, *data.shape[2:]))

        bin_arrays = [data, data[1:,:,:,:], data[:,1:,:,:], data[1:,1:,:,:]]

        for i, d in enumerate(bin_arrays):
            shape = d.shape[:2]

            # the size the data will be reduced to #
            x_reduced, x_rest = shape[0]//bin_size[0], shape[0]%bin_size[0]
            y_reduced, y_rest = shape[1]//bin_size[1], shape[1]%bin_size[1]


            if x_rest != 0: #odd dimension
                d = d[:-1,:]

            if y_rest != 0: #odd dimension
                d = d[:,:-1]

            shape = np.array(d.shape[:2], dtype=int)
            new_shape = shape//np.array(bin_size, dtype=int)

            # bin the data #
            binned = d.reshape((new_shape[0], shape[0]//new_shape[0],
                        new_shape[1], shape[1]//new_shape[1],
                                *d.shape[2:])).mean(-3).mean(1)

            if i==0:
                new[::2,::2,:,:] = binned
            elif i==1:
                new[1::2,::2,:,:] = binned
            elif i==2:
                new[::2,1::2,:,:] = binned
            elif i==3:
                new[1::2,1::2,:,:] = binned

        self.binned_data = new
        
    def reconstruct_adf(self, detector_type, radii=(35, None), binning=False):
        
        if detector_type == 'maadf':
            inner_radius = radii[0]    
            if radii[1]==None:
                radii = inner_radius*1e-3/self.scale, self.dp_shape[0]//2 #use largest possible
            else:
                radii = inner_radius*1e-3/self.scale, radii[1]*1e-3/self.scale

            if binning==False:
        
                virtual_image = self.datacube.get_virtual_image(mode = 'annulus',
                                                           geometry = (self.center, radii),
                                                           name = 'annular_dark_field',
                                                           verbose=True)
                self.virtual_image = virtual_image

            elif binning==True:
                data = py4DSTEM.DataCube(self.binned_data)
                virtual_image = data.get_virtual_image(mode = 'annulus',
                                                           geometry = (self.center, radii),
                                                           name = 'annular_dark_field',
                                                           verbose=True)
                self.virtual_image_binned = virtual_image

    def show_images(self, gaussian=3, save=False):
        images = [self.real_image, self.virtual_image.data]
        names = ['real_image', 'virtual_image']

        for i, im in enumerate(images):
            if names[i] == 'virtual_image':
                factor = self.virtual_image.shape[1]/self.real_image.shape[1] #for the scalebar
                gaussian=0
            else:
                factor = 1
                
            im_scale = self.image_scale/factor
            
            fig, axs = hf.make_scalebar(im, im_scale, (1,2,(1,2)), unit='nm')
            axs[0].imshow(skimage.filters.gaussian(im, sigma=gaussian).transpose(), cmap='plasma')
            axs[1].imshow(skimage.filters.gaussian(im, sigma=gaussian).transpose(), cmap='gray')
            if save:
                savepath = save + '/' + names[i] + '.pdf'
                plt.savefig(savepath, bbox_inches='tight', pad_inches=0)

    def visualize_all(self, mask_radius=100, gaussian=1, save=False):
        
        path = save
        
        ## mean diff pattern ##

        fig, axs = hf.make_scalebar(self.dp_mean, self.scale, (1,1,1), unit='mrad')
        py4DSTEM.show(self.dp_mean.transpose(), scaling='log', vmin=0, vmax=.95, cmap='plasma', 
                      figax=(fig,axs))
        if save:
            path = save + '/mean_diff_pattern.pdf'
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
        
        ## example image ##
        fig, axs = hf.make_scalebar(self.dp_mean, self.scale, (1,1,1), unit='mrad')
        axs.imshow(skimage.filters.gaussian(hf.mask(self.datacube[1,1].compute()-self.dp_mean, 
                                                       mask_radius), sigma=gaussian).transpose(), 
                            cmap='plasma')
        if save:
            path = save + '/example_diff_pattern.pdf'
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
        
        
        ## convergence angle ##
        if save:
            path = save + '/conv_angle.pdf'
        thresholded = hf.threshold(self.dp_mean, self._conv_params[0], 
                                   gaussian=self._conv_params[1], mask_radius=False, 
                                   verbose=self.verbose, scale=self.scale, save=path)
        
        ## binary image ##
        summed, bin_image = self._bin_image_steps
        fig, axs = hf.make_scalebar(summed, self.scale, (1,3,(1,3)), unit='mrad')

        axs[1].hist(summed.ravel(), bins=256, density=True);
        axs[0].imshow(summed.transpose(), cmap='plasma')
        axs[2].imshow(bin_image.transpose(), cmap='gray')
        plt.tight_layout()
        
        if save:
            path = save + '/binary_image.pdf'
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
        
        
        ## center points and angles ##
        
        fig, axs = hf.make_scalebar(bin_image, self.scale, (1,1,1), unit='mrad')
        axs.imshow(bin_image.transpose(), cmap='plasma')

        circle = plt.Circle(self.center, self.r_outer, edgecolor='red', facecolor=None, 
                                fill=False, ls='--')
        
        centroids = np.array([r['centroid'] for r in self._all_regions])
        axs.plot(centroids[:,0], centroids[:,1], 'x')
        axs.add_patch(circle)
        
        if save:
            path = save + '/center_point_image.pdf'
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
        
        ## label image ##
        fig, axs = hf.make_scalebar(self.label_image, self.scale, (1,1,1), unit='mrad')
        axs.imshow(hf.mask(self.label_image, mask_radius).transpose(), cmap='plasma')
        
        #vectors = self.vectors
        regions = list(sum(self.pairs, ()))
        
        for i, c in enumerate([r['centroid'] for r in regions]):
            axs.plot(c[0], c[1],'x', color='maroon')
       
        if save:
            path = save + '/label_image.pdf'
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
            
    def fit_position(self, position, ps, gaussian=2, rotate_avrg=False, binning=False, silent=False, save=False):
        # if you want to fit a single position #
        if binning==True:
            if type(self.binned_data)!=type(None):
                pattern = self.binned_data[position[0], position[1]]
                
            else:
                pattern = np.mean([self.datacube[position[0], position[1]], 
                               self.datacube[position[0], position[1]+1],
                               self.datacube[position[0]+1, position[1]],
                               self.datacube[position[0]+1, position[1]+1]], axis=0)      
        else:
            pattern = self.datacube[position[0], position[1]]

        if rotate_avrg==True:
            angles = [self.orientation + i*np.pi/3 for i in range(6)]
            rotated_copies = [skimage.transform.rotate(pattern, np.degrees(a), center=self.center)\
                    for a in angles]
            pattern = np.mean(rotated_copies, axis=0)
        
        pattern = pattern - self.dp_mean

        dp = DP(pattern, disk_mask=self.disk_mask, 
                    gaussian=gaussian, scale=self.scale, center=self.center,
                    orientation=self.orientation, radius=self.r_outer, verbose=self.verbose)
        plt.show()
        if save:
            before = dp.verbose
            dp.verbose=True
            dp.prepare_disks(save)
            dp.verbose = before
        
        ## fit the position ##
        dp.position_fitter(ps, save=save, silent=silent)

        return dp
    
    def analyze_map(self, ps, savedir, gaussian=2, rotate_avrg=False, positions=False, binning=False):
        # if you want to fit the whole map #
        if binning==True:
            shape = (self.datacube.shape[0]-1, self.datacube.shape[1]-1)        
        else:
            shape = self.datacube.shape[:2]
        
        if positions == False: 
            comparison_array = np.ones(shape)
            x_idx, y_idx = np.where(comparison_array)
            positions = [[x_idx[i], y_idx[i]] for i in range(len(x_idx))] # use all positions

        spots = []

        savefile = savedir + 'fitting_parameters.txt'
        for i in tqdm(range(len(positions[::2]))): #only use every second position
            position = self.fit_position(positions[i], ps, 
                                         gaussian=gaussian, rotate_avrg=rotate_avrg, binning=binning, 
                                         silent=True)
    
            with open(savefile, 'a') as f:
                for d in position.disks:
                    if d.success==True:
                        f.write(str(d.p0)[1:-1]+' '+ str(d.vector[0]) + \
                                ' ' + str(d.vector[1]) + ' ' + str(d.errors)[1:-1] +'\n')
                    else: 
                        f.write('0\n')
                f.write('\n')
                
        defocus_array, defocus_mean, \
                ild_array, ild_std,ild_mean, rotation_array, \
                A_array, phi_array, lambda_array = self.get_fitted_maps(savefile, save=savedir)
        
        return defocus_array, defocus_mean, \
                    ild_array, ild_mean, rotation_array, \
                        A_array, phi_array, lambda_array
                
    def get_fitted_maps(self, savefile, diskslice=slice(0,6), save=False):
        # analyze the whole fitted map #

        name = f'_slice_{diskslice.start}_{diskslice.stop}.txt'
    
        ia = InterferometricAnalyzer(self)
        
        defocus_array = []
        defocus_mean = None

        ild_array = []
        ild_std = []
        ild_mean = None

        rotation_array = []

        A_array = []
        phi_array = []
        lambda_array = []

        with open(savefile, 'r') as f:
            content = f.read()

        all_disks = []

        #get list of all the content for every position in file
        position_list = content.split('\n\n')[:-1]
        position_list = [p.split('\n') for p in position_list]

        # next get a list where on every position there are three 
           #sublists each containing A, phi, nu, dir_angle, vector to disk x, 
                #vector to disk y, error A, error phi, error nu, error dir_angle
                
        p0_array = []
        for i, p in enumerate(position_list[:-1]):
            disks = []
            for d in p:
                if d=='0':
                    continue
                else:
                    disks.append([float(n) for n in d.split(' ') if n])
            p0_array.append(disks)
            # now we have an array containing a list of lists of parameters
                # for the disks of every position

        ## interlayer distance ##

        for l in p0_array[:]:
            position_ild = []
            position_rotation = []

            for p0 in l[diskslice]:


                vector = np.array([p0[4], p0[5]])
                dir_angle = p0[3]
                nu = p0[2]
                
                rot = ia.get_rotation(vector, dir_angle, self.orientation)
                position_rotation.append(rot)
                position_ild.append(ia.get_ild(rot, nu))

        
            # save means of this position #
            rotation_array.append(np.mean(position_rotation))
            ild_array.append(np.mean(position_ild))
            ild_std.append(np.std(position_ild))

        shape = (int(np.sqrt(len(p0_array))),
                int(np.sqrt(len(p0_array))))

        rotation_array = np.array(rotation_array).reshape(shape)
        ild_array = np.array(ild_array).reshape(shape)
        ild_std = np.array(ild_std).reshape(shape)

        rotation_array = np.where(np.degrees(rotation_array) < 30, rotation_array, np.nan)
        ild_array = np.where(np.degrees(rotation_array) < 30, ild_array, np.nan)
        ild_std = np.where(np.degrees(rotation_array) < 30, ild_std, np.nan)

        ild_mean = np.mean(ild_array)

        if save:
            np.savetxt(save + '/rotation_array'+name, rotation_array)
            np.savetxt(save + '/ild_array'+name, ild_array)

        # get array of defocus #
        for l in p0_array:
            position_defocus = [] #list of defocus for every disk

            for p0 in l[diskslice]:
                position_defocus.append(ia.get_defocus(ia.get_fringe_wavelength(p0[2])))

            # save the mean value of position #
            defocus_array.append(np.mean(position_defocus))

        defocus_array = np.array(defocus_array).reshape(shape)
        defocus_array = np.where(np.degrees(rotation_array) < 30, defocus_array, np.nan)


        if save:
            path = save + '/defocus_array'+name
            np.savetxt(path, defocus_array)

        defocus_mean = np.mean(defocus_array)

        ## other stuff ##

        for l in p0_array:
            position_A = []
            position_phi = []
            position_lambda = []

            for p0 in l[diskslice]:
                position_A.append(p0[0])
                position_phi.append(p0[1])
                position_lambda.append(ia.get_fringe_wavelength(p0[2]))

            A_array.append(np.mean(position_A))
            phi_array.append(np.mean(position_phi))
            lambda_array.append(np.mean(position_lambda))

        A_array = np.array(A_array).reshape(shape)
        phi_array = np.array(phi_array).reshape(shape)
        lambda_array = np.array(lambda_array).reshape(shape)

        A_array = np.where(np.degrees(rotation_array) < 30, A_array, np.nan)
        phi_array = np.where(np.degrees(rotation_array) < 30, phi_array, np.nan)
        lambda_array = np.where(np.degrees(rotation_array) < 30, lambda_array, np.nan)

        if save:
            np.savetxt(save + '/A_array'+name, A_array)
            np.savetxt(save + '/phi_array'+name, phi_array)
            np.savetxt(save + '/lambda_array'+name, lambda_array)

        return defocus_array, defocus_mean, \
                    ild_array, ild_std, ild_mean, rotation_array, \
                        A_array, phi_array, lambda_array

    def analyze_disks(self, disks):
        # method to analyze one or multiple disks #
        ia = InterferometricAnalyzer(self)
        
        defocus_list = []
        ild_list = []
        rotation_list = []
        ild_error_list = []
        nu_error_list = []
        nu_list = []
        
        
        for d in disks:
            if d.success==True:
                defocus_list.append(ia.get_defocus(ia.get_fringe_wavelength(d.nu)))
                rotation = ia.get_rotation(d.vector, d.dir_angle, self.orientation)
                rotation_list.append(rotation)
                ild_list.append(ia.get_ild(rotation, d.nu))
                ild_error_list.append(d.errors[-1])
                nu_list.append(abs(d.nu))
                nu_error_list.append(d.errors[-2])
            else:
                rotation_list.append(np.nan)
                ild_list.append(np.nan)
                ild_error_list.append(np.nan)
                nu_list.append(np.nan)
                nu_error_list.append(np.nan)
            
        defocus = np.mean(defocus_list)
        ild = np.mean(ild_list)
        
        return defocus_list, ild_list, rotation_list, defocus, ild, ild_error_list, nu_list, nu_error_list
    
class DP:
    """ 
    class representing a diffraction pattern of a DataAnalyzer object
    
    Attributes:
    ----------------------
    dp (array)                          :   diffraction pattern
    verbose (bool)                      :   if True, more steps are shown
    disks (list)                        :   list of the three averaged diffraction disks
    gaussian (float or bool)            :   if float uses a gaussian filter with it as sigma
    disk_mask (float)                   :   mask size in mrad of the bragg disks
    scale (float)                       :   the scale in rad/px of the dp
    orientation (float)                 :   orientation in rad of the hexagon corresponding to 
                                            the centers of the bragg disks of the armchair planes
                                            away from the x-axis
    radius (float)                      :   the radius of the ac bragg disks in px
    center (tuple)                      :   the coordinates of the dp center

    Methods:
    ----------------------
    avrg_and_mask                       :   masks, filters the disks    
    prepare_disks                       :   method to find the disks and pairs, as well as the vectors
                                            pointing to the disks
    disk_fitter                         :   fits a disk if the data and vector is provided 
    position_fitter                     :   fits the disks corresponding to this diffraction pattern 
    plot_disks                          :   plots the diffraction disks
    position_fitter                     :   fits all disks of one position
    generate_p0                         :   creates new starting conditions
    get_stats                           :   calculates the statistics of the fit

    """
    
    def __init__(self,  dp, disk_mask, scale, orientation, radius, center, verbose=False, gaussian=2):
        
        self.dp = dp # already substracted mean before passing to function 
        self.gaussian = gaussian
        self.verbose = verbose
        self.disk_mask = disk_mask
        self.scale = scale
        self.orientation = orientation
        self.radius = radius
        self.center = center
        
        self.disks = self.prepare_disks(avrg=avrg)
        
    def average_and_mask(self, im, spot_centers):

        # scale the radius from mrad to px #
        scaled_radius = int(self.disk_mask*1e-3/self.scale)

        # create mask #
        mask = np.zeros((scaled_radius*2, scaled_radius*2), dtype=bool)
        rr, cc = skimage.draw.disk((scaled_radius, scaled_radius), scaled_radius)
        mask[rr, cc] = True
        
        disks = []
        vectors = []
        
        for c in spot_centers:

            start_x = int(c[0]-scaled_radius) 
            stop_x = int(c[0]+scaled_radius) 
            start_y= int(c[1]-scaled_radius)
            stop_y = int(c[1]+scaled_radius)

            # crop to the calculated start and stop
            cropped = im[start_x:stop_x, start_y:stop_y]
            
            # get the seperate disks #
            disks.append(cropped)
            vectors.append(hf.unit_vector(np.array(c-self.center))) 

        # filter and mask disks #
        filtered_disks = [np.where(mask, skimage.filters.gaussian(d, sigma=self.gaussian), 
                                    0)[1:,1:]\
                                 for d in disks]

        return filtered_disks
    
    def prepare_disks(self, save=False):
        
        all_disks = []
        all_vectors = []

        spot_centers = hf.hexagon(self.orientation, self.radius, self.center)
        
        vectors = [hf.unit_vector(np.array(c-self.center)) for c in spot_centers]

        disks = self.average_and_mask(self.dp, spot_centers)

        
        if self.verbose==True:
            fig, axs = hf.make_scalebar(disks[0], self.scale, (3,2,(1,2,3,4,5,6)),
                                        unit='mrad')
            axs = axs.reshape((3,2))
            for i in range(3):
                axs[i, 0].imshow(disks[i].transpose(), cmap='plasma')
                axs[i, 1].imshow(disks[i+3].transpose(), cmap='plasma')
                axs[i, 0].set_title(f'disk {i}')
                axs[i, 1].set_title(f'disk {i+3}')

            if save:
                path = save + f'/all_disks.pdf'
                plt.savefig(path,  bbox_inches='tight', pad_inches=0)

            plt.show()


        return [Disk(disks[i], scale=self.scale, vector=vectors[i]) for i in range(len(disks))]

    def disk_fitter(self, ps, disk_data, vector, silent=False, save=False):
        # to fit specific data only - mostly for bug fixing and tryouts #
        
        d = Disk(disk_data, self.scale, vector, p0=ps)
        
        j = 0 #counter for tries

        rs_s = [] #empty array to be filled with r_square values
        p0_s = [] #empty array to be filled with p0 values

        iterator = iter(ps) 

        d.verbose = self.verbose
        d.gaussian = self.gaussian

        while True:
            # do as long as either works or max iteration is reached #
            j += 1
            if j == 20:
                if silent==False:
                    print("Reached the 20th iteration - I give up.")
                d.p0 = None
                d.success = False #set status to failed
                break

            if self.verbose==True:
                print(f"------ try {j} ------")

            try:
                p0 = next(iterator)

            except StopIteration:
                # if we run out of given starting values #
                p0 = self.generate_p0(np.array(rs_s), p0_s)

            d.p0 = p0
            
            try: # fit the disk 
                popt, r_squared, errors = d.fit_cosine()
                p0_s.append(p0)
                rs_s.append(r_squared)

                if 1.1 > r_squared > .85:
                    # the fit is successful #
                    d.success = True
                    d.p0 = popt
                    d.r_squared = r_squared
                    d.errors = errors
                    break

            except RuntimeError:
                # the fit is not successful #
                d.success = False
                d.p0 = None
                break

        if d.success==True and self.verbose==False:
            if silent==False:
                print(f"r² = {round(d.r_squared, 4)}")
                fit_image = d.generate_image(d.p0).reshape(d.shape)
                d.plot_comparison(d.p0, save=save)
                plt.show()
                
        return d
        
    def position_fitter(self, ps, disk_slice=None, rotate_avrg=False, silent=False, save=False):
        # fit one position #
        
            # iterate over all disks #
        if disk_slice==None:
            disk_slice = slice(0, len(self.disks))

        for i, d in enumerate(self.disks[disk_slice]):
            
            if silent==False:
                print(f"====== disk {i} ======")
                
            j = 0 #counter for tries
            
            rs_s = [] #empty array to be filled with r_square values
            p0_s = [] #empty array to be filled with p0 values
            
            iterator = iter(ps) 
            
            d.verbose = self.verbose
            d.gaussian = self.gaussian
            
            while True:
            # do until fit is either successful or max iterations are reached #
                j += 1
                if j == 10:
                    if silent==False:
                        print("Reached the 10th iteration - I give up.")
                    d.p0 = None
                    d.success = False #set status to failed
                    break
                    
                if self.verbose==True:
                    print(f"------ try {j} ------")
                
                try:
                    p0 = next(iterator)
                    
                except StopIteration: # generate new starting condition 
                    p0 = self.generate_p0(np.array(rs_s), p0_s)
                    
                d.p0 = p0
                
                try:
                    popt, r_squared, errors = d.fit_cosine()
                    p0_s.append(p0)
                    rs_s.append(r_squared)
                    
                    if 1.1 > r_squared > .85:
                        # fit successful #
                        d.success = True
                        d.p0 = popt
                        d.r_squared = r_squared
                        d.errors = errors
                        break
                    
                except RuntimeError:
                    # fit failed #
                    d.success = False
                    d.p0 = None
                    break
                    
            
            if d.success==True and self.verbose==False:
                if silent==False:
                    print(f"r² = {round(d.r_squared, 4)}")
                    fit_image = d.generate_image(d.p0).reshape(d.shape)
                    d.plot_comparison(d.p0, save=save)
                    plt.show()
                    
    def generate_p0(self, rs_s, p0_s):
        
        # how off is the r squared value? #
        dr = abs(1-rs_s)

        # use better of last two tries #
        if dr[-1] < dr[-2]:
            original = np.array(p0_s[-1])
        else: 
            original = np.array(p0_s[-2])

        l = len(original)
        # go random step scaled by the last good try #
        rando = (1 - 2*np.random.rand(l)) * original

        new = original + rando
        
        return new
    
    def get_stats(self, save=False):
        # return stats for disks of this pattern #
        
        all_disks = ''
        
        for i, d in enumerate(self.disks):
            if d.success==True:
                disk = f'disk {i}\n\n'
                stats = f'A : {d.A}\n' + \
                        f'phi : {d.phi}\n' + \
                        f'nu : {d.nu}\n' + \
                        f'dir_angle : {d.dir_angle}\n' +\
                        f'wl : {1/(d.nu) * self.scale} rad\n'  +\
                        f'r_squared : {d.r_squared}\n' +\
                        f'dir_error : {d.errors[-1]}'
            else:
                continue
            
            all_disks += str(disk + stats + '\n\n\n')
            
        print(all_disks)
        if save:
            path = save + f'/disk_params.txt'
            with open(path, 'w') as f:
                f.write(all_disks)

class Disk:
    """
    A class for fitting cosinus disks.
    
    Attributes:
    ----------------------
    data (array)            :   data corresponding to the disk
    scale (float)           :   the scale of the images in rad/px
    vector (array)          :   unit vector in direction of disk from center
    shape (tuple)           :   shape of the disk
    success (bool)          :   True if the fit worked, False if not, None if not attempted yet
    p0 (tuple)              :   contains the fit parameters (A, phi, nu, dir_ang)
    fit_image (array)       :   the image constructed from the fit parameters
    A (float)               :   amplitude of the cosine
    phi (float)             :   phase shift in px
    nu (float)              :   frequency in 1/px
    dir_angle (float)       :   angle of the cosine direction in rad
    n (array)               :   unit vector corresponding to dir_angle
    r_squared (float)       :   the r^2 value of the fit
    errors (list)           :   list containing the fitting uncertainties
    verbose (bool)          :   if True, more steps are shown

    Methods:
    ---------------------
    cosine                  :   basic 2d cosine function used for fitting
    generate_image          :   generates a masked image from the cosine function
    fit_cosine              :   fits the cosine to the data
    plot_with_errors        :   plots a fit with p0-errors, p0 and p0+errors
    plot_comparison         :   plots a detailed comparison of data and fit
    calculate_statistics    :   calculates fitting statistics
    """
    
    def __init__(self, data, scale=None, vector=None, p0=None, verbose=False):
        self.data = data
        self.scale = scale
        self.vector = vector
        self.shape = data.shape
        self.success = None
        self.p0 = p0

        self.fit_image = None
        
        ## components of p0 ##
        self.A = None
        self.phi = None
        self.nu = None
        self.dir_angle = None
        self.n = None
        ## --------------- ##
        
        self.r_squared = None
        self.errors = None
        self.verbose = verbose
    
    @property
    def p0(self):
        return self._p0
    
    @p0.setter
    def p0(self, value):
        self._p0 = value
        if self.success==True:
            self.A = value[0]
            self.phi = value[1]
            self.nu = value[2]
            self.dir_angle = value[3]
            self.n = np.array([np.sin(self.dir_angle), np.cos(self.dir_angle)])

            x = np.arange(0, self.shape[0])
            y = np.arange(0, self.shape[1])
            xx, yy = np.meshgrid(x,y)
            self.fit_image = self.cosine((xx,yy),
                    self.A, self.phi, self.nu, self.dir_angle).reshape(self.shape)
    
    def cosine(self, xy, A, phi, nu, dir_angle):
        # creates masked cosine function #

        x, y = xy
        center = (self.shape[0]//2, self.shape[1]//2)

        #x, y = np.linspace(0,1,self.shape[0]), np.linspace(0,1,self.shape[1])
        #xx, yy = np.meshgrid(x,y)

        n1, n2 = np.cos(dir_angle), np.sin(dir_angle)

        #nu = 44308.51656642421 #try this value
        k1 = 2*np.pi*nu*np.cos(dir_angle)
        k2 = 2*np.pi*nu*np.sin(dir_angle)
        
        array = A*np.cos((x*k1 + y*k2) + phi)

        #array = A*np.cos(2*np.pi*nu*(x*n1 + y*n2) + phi)
        mask = np.zeros((max(self.shape), max(self.shape)), dtype=bool)
        rr, cc = skimage.draw.disk(
            ((self.shape[0]-1)//2, (self.shape[1]-1)//2),
            max(self.shape)//2+1)

        mask[rr, cc] = True
        return np.where(mask, array, 0).ravel()
        
    def generate_image(self, popt):
        # generates flattened cosine from set of parameters #

        x = np.arange(0, self.shape[0])
        y = np.arange(0, self.shape[1])

        xx, yy = np.meshgrid(x, y)
        image = self.cosine((xx, yy), *popt).reshape(self.shape)
        return image.ravel()
    
    def fit_cosine(self):
        # fits a cosine to the data of the disk #

        # Create x and y indices
        x = np.arange(0, self.shape[0])
        y = np.arange(0, self.shape[1])
        x, y = np.meshgrid(x,y)

        #create data from real image
        data = self.data.ravel()

        if self.verbose==True:
            self.plot_comparison(self.p0)

        ## fit the cosine function ##
        
        popt, pcov = opt.curve_fit(self.cosine, (x, y), data, p0=self.p0, maxfev=5000)
        errors = np.sqrt(np.diag(pcov))
        if self.verbose==True:
            self.plot_comparison(popt)
            plt.show()

        residual, irrelevant1, irrelevant2, r_squared = self.calculate_statistics(data.reshape(self.shape), popt)

        if self.verbose==True:
            print(r_squared)


        return popt, r_squared, errors

    def plot_with_errors(self, save=False):
        fig, axs = hf.make_scalebar(self.data, self.scale, (1,4,(1,2,3,4)), unit='mrad')
        
        fit = self.generate_image(self.p0).reshape(self.shape)
        fit_u1 = self.generate_image(self.p0+self.errors).reshape(self.shape)
        fit_u2 = self.generate_image(self.p0-self.errors).reshape(self.shape)

        axs[0].imshow(self.data.transpose(), cmap='plasma')
        axs[1].imshow(fit.transpose(), cmap='plasma')
        axs[2].imshow(fit_u1.transpose(), cmap='plasma')
        axs[3].imshow(fit_u2.transpose(), cmap='plasma')

        axs[0].set_title('data')
        axs[1].set_title('fit')
        axs[2].set_title('fit + uncertainties')
        axs[3].set_title('fit - uncertainties')

        plt.show()

    def plot_comparison(self, popt, save=False):
        
        fig, (ax0, ax1, ax2, ax3) = hf.make_scalebar(self.data, self.scale, (1,4,(1,2)), unit='mrad')
        
        ax2.remove()
        ax3.remove()
        
        ax2 = fig.add_subplot(143, projection='3d')
        ax3 = fig.add_subplot(144, projection='3d')
        
        data = self.data
            
        fit = self.generate_image(popt).reshape(self.shape)
            
        ax0.imshow(data.transpose(), cmap='plasma')
        ax0.set_title('disk')

        ax1.imshow(fit.transpose(), cmap='plasma')
        ax1.set_title('fit')

        x = np.arange(0, self.shape[0])
        y = np.arange(0, self.shape[1])

        xx, yy = np.meshgrid(x, y)

        ax2.plot_surface(xx, yy, data, cmap=cm.Blues, linewidth=0, alpha=.5)
        ax2.plot_surface(xx, yy, fit, cmap=cm.Reds, linewidth=0,  alpha=.5)
        ax2.set_title('surface plots: data: blue, fit: red')

        ddisk = data - fit

        ax3.plot_surface(xx, yy, ddisk, cmap=cm.coolwarm, linewidth=0)
        ax3.set_title("difference")
        
        if save:
            path = save + f'/disk_fit_{np.round(np.random.random()*1000)}.pdf' 
            plt.savefig(path,  bbox_inches='tight', pad_inches=0)

    def calculate_statistics(self, disk, popt):
        # calculates the fitting errors #

        expected = self.generate_image(popt).reshape(self.shape)
        residual = disk - expected
        ss_res = np.sum(residual**2)
        ss_tot = np.sum((disk-np.mean(disk))**2)
        r_squared = 1 - (ss_res / ss_tot)
        return residual, ss_res, ss_tot, r_squared

class InterferometricAnalyzer:
    """
    A class to take a fitted cosinus function and decern different physical things from it.

    Attributes:
    -------------------------
    twist_angle             :   twist angle of the region in radians
    scale                   :   scale in rad/px
    energy                  :   energy of map acquisition in keV
    bragg_angle             :   Bragg angle corresponding to armchair spots

    Methods:
    -------------------------
    get_fringe_wavelength   :   calculates the fringe wavelength from nu
    get_wavelength_from_defocus:calculates the fringe wavelength from a given defocus
    get_defocus             :   calculates the defocus from the given values
    get_rotation            :   determines the fringe rotation
    get_ild                 :   calculates the ild in m
    get_ild_errors          :   calculates the error in ild using Gaussian error propagation
    """
    
    def __init__(self, data_object):
        self.twist_angle = data_object.twist_angle #in RADIANS!
        self.scale = data_object.scale
        self.energy = data_object.energy
        
        a = 2.46 #in angstrom
        d_zz = a*np.sin(np.pi/3)
        d_ac = a/2
        self.bragg_angle = hf.bragg_angle(d_ac, self.energy) #we use the outer spots - armchair planes
        
    def get_fringe_wavelength(self, nu):
        """
        Calculates the fringe wavelength from the fitted frequency.
        Args:
            nu (float)    :    fitted frequency in 1/px
        Returns:
            wl (float)    :    wavelength of interference fringes in rad
        """
        
        wl = 1/(nu) * self.scale
        return abs(wl)
    
    def get_wavelength_from_defocus(self, defocus):
        
        defocus = defocus*1e-10

        U = self.energy*1e3*e #to joule
        
        # calculate electron wavelength #
        _lambda_el = h/np.sqrt( 2*m_e*U * (1 + U/(2*m_e*c**2)))

        fringe_wavelength = _lambda_el/(defocus*np.sin(self.twist_angle/2)*4*self.bragg_angle)
        nu = 1/(fringe_wavelength*self.scale)

        return fringe_wavelength, nu

    def get_defocus(self, fringe_wavelength):
        """
        Calculates the defocus from the fringe wavelength of the interference disks. 
        Args:
            fringe_wavelength (float)    :    interference fringe wl in rad
        Returns:
            defocus (float)              :    the defocus of the position in m
        """
        U = self.energy*1e3*e #to joule
        
        # calculate electron wavelength #
        _lambda_el = h/np.sqrt( 2*m_e*U * (1 + U/(2*m_e*c**2)))

        defocus = _lambda_el / (4 * self.bragg_angle * fringe_wavelength * np.sin(self.twist_angle/2)) 
                #not sure about factor 4 here

        return defocus
    
    def get_rotation(self, disk_vector, dir_angle, orientation):
        """
        To calculate the rotation angle of the pattern wrt the orientation towards
        the center we use the vector calculated before of the disk connecting to 
        the center of the pattern and the vector of the cosinus function n.
        n points in the direction of propagation, so we need the perpendicular one.
        
        Args:
            disk_vector (array)    :    unit vector pointing in the disk direction
            dir_angle (float)      :    angle in direction of the cosine propagation in rad
            orientation (float)    :    orientation of the hexagon of the dp spots fitted above in rad
        Returns:
            angle (float)          :    rotation from neutral position of disk to real position
        """
        
        n = np.array([np.sin(dir_angle), np.cos(dir_angle)])
        perp = np.array([-n[1], n[0]]) #create perpendicular vector

        disk_v = disk_vector
        
        angle = hf.angle_between(disk_v, perp)# perp)
        return angle
    
    def get_ild(self, rotation, nu):
        """
        Calculate the interlayer distance from the rotation of the fringes and the 
        fringe wavelength (or frequency).
        
        Args:
            rotation (float)    :    rotation of disk interference fringes in rad
            nu (float)          :    fitted fringe frequency in 1/px
        Returns:
            ild (float)         :    interlayer distance in m
        """
        
        U = self.energy*1e3*e #to joule
        _lambda_el = h/np.sqrt( 2*m_e*U * (1 + U/(2*m_e*c**2)))
        
        fringe_wl = self.get_fringe_wavelength(nu)
        
        ild = _lambda_el*np.sin(rotation) / (fringe_wl*np.tan(2*self.bragg_angle)*np.cos(self.twist_angle/2)) 

        return ild
    
    def get_ild_error(self, rot, nu, drot, dnu):
        """
        Calculate the error in interlayer distance from the rotation of the fringes
        the fringe frequency and the errors of the two.

        Args:
            rot (float)         :   rotation of the disk interference fringes in rad
            nu (float)          :   frequency of the fringes in 1/px
            drot (float)        :   fit uncertainty of the rotation in rad
            dnu (float)         :   fit uncertainty of the fringe frequency in 1/px
        Returns:
            d_ild (float)       :   the uncertainty of the interlayer distance in m
        """
        U = self.energy*1e3*e #to joule
        _lambda_el = h/np.sqrt( 2*m_e*U * (1 + U/(2*m_e*c**2)))
        
        const = _lambda_el/(np.tan(2*self.bragg_angle)*np.cos(self.twist_angle/2))
        wl = self.get_fringe_wavelength(nu)
        
        d_ild = const*(np.cos(rot)*drot/wl + np.sin(rot)*dnu/nu)
        return d_ild

     
