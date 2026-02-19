import numpy as np
from tensorflow.keras.optimizers import SGD
import sys
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.util import img_as_uint
import cv2
#import pyfits
from astropy.io import fits
import sunpy.map
from aiapy.calibrate import register


class t_cnn_predictor(object):


    def save_data_png_ri(self, data, fname ):
    
        img_arr = exposure.rescale_intensity(data, out_range='float')
        img_arr = img_as_uint(img_arr)
        cv2.imwrite(fname, img_arr, [cv2.IMWRITE_PNG_COMPRESSION, 0]) #  No compression 
    
    

    def window_filt(self, data,  boxsize, increment, thrhold, size_x, size_y): 
           
        result_data = np.copy(data)
           
           
        for x_1 in range(1,size_x-boxsize, increment):
            for y_1 in range(1,size_y-boxsize, increment): 
               
                x_m=x_1+boxsize
                y_m=y_1+boxsize 

                border_horz_l=data[x_1:x_m, y_1:(y_1+1)]
                border_horz_r=data[x_1:x_m, y_m:(y_m+1)]
                
            
                border_vert_l=data[x_1:(x_1+1), y_1:y_m]
                border_vert_r=data[x_m:(x_m+1), y_1:y_m]

                border_horz_l_max=np.amax(border_horz_l)
                border_horz_r_max=np.amax(border_horz_r)
           
                border_vert_l_max=np.amax(border_vert_l)
                border_vert_r_max=np.amax(border_vert_r)
               
                border_horz_l_min=np.amin(border_horz_l)
                border_horz_r_min=np.amin(border_horz_r)
           
                border_vert_l_min=np.amin(border_vert_l)
                border_vert_r_min=np.amin(border_vert_r)

                area_mean = np.mean(result_data[x_1+1:x_m-1, y_1+1:y_m-1])
                  
                borders_mean =  (np.mean(border_horz_l)+np.mean(border_horz_r)+
                              np.mean(border_vert_l)+np.mean(border_horz_r)) /4

            
                if ((border_horz_l_max<=thrhold ) and (border_horz_r_max<=thrhold) and (border_vert_l_max<=thrhold) and (border_vert_r_max<=thrhold)):              
                 
                    result_data[x_1:x_m, y_1:y_m] = 0 #np.zeros( (boxsize,boxsize), dtype = type(data))
                    
            
                               
                elif ((border_horz_l_min>thrhold ) and (border_horz_r_min>thrhold) and (border_vert_l_min>thrhold) and (border_vert_r_min>thrhold)):              
                    
                   #  replace for loop with multiplyin matrixes

                    for x_i in range(x_1+1,x_m-1, 1):
                       for y_i in range(y_1+1,y_m-1,1):
                           if (result_data[x_i,y_i]<=thrhold):
                               result_data[x_i,y_i]=borders_mean#area_mean
                
            
        return result_data


    
    def normalize_0_1(self,data_arr):   

        data_arr_no_nan = np.nan_to_num(data_arr)
        data_arr_min = np.amin(data_arr_no_nan) 
        data_arr_from_zero = np.subtract(data_arr_no_nan,data_arr_min)
        data_arr_max = np.amax(data_arr_from_zero) 
        if (data_arr_max!=0):
            data_arr_adj_f = np.multiply(data_arr_from_zero,1/data_arr_max)
            return data_arr_adj_f
        else: 
            return np.zeros_like(data_arr)
            

    def load_folder_data(self, path ):
       
        files_arr = os.listdir(path)
        files_arr.sort()
         
        print('Loading numpy data from:'+path,'...')
        files_n = len(files_arr)
        self.data_frames_n = min(self.max_files_n, files_n) 
        self.data=np.empty([self.data_frames_n,self.channels_n,self.size_x,self.size_y],dtype=float)
    
        for file_i in range(1,files_n-1):
                      
            
            full_path = path+'/'+files_arr[file_i]
            print('Reading data from:' +full_path)  

            if (( full_path.find('fits') != -1) or ( full_path.find('fts') != -1) ):
                   
                print('Loading fits data...')
                
                img_init_lev_1=sunpy.map.Map(path+'/'+files_arr[file_i])
                #img_init_lev_1_5 = register(img_init_lev_1)
                #helio_data=np.array(img_init_lev_1_5.data) 
                helio_data=np.array(img_init_lev_1.data) 
                
                '''
                try:
                    
                    img_init_lev_1=sunpy.map.Map(path+'/'+files_arr[file_i])
                    img_init_lev_1_5 = register(img_init_lev_1)
                    helio_data=np.array(img_init_lev_1_5.data)
                    #helio_data=np.array(img_init_lev_1.data)

                except: 
                    sys.exit("Failed to load FITS data.")

                '''
            else:
             
                print('Loading numpy data.... ')        
                
                try: 
                    
                    helio_data=np.load(path+'/'+files_arr[file_i])
                
                except:
                    sys.exit("Failed to load numpy data.")
        
            helio_data = exposure.rescale_intensity(helio_data, out_range='float')
            helio_data = img_as_uint(helio_data)
            helio_data = cv2.resize(helio_data, dsize=(self.size_x, self.size_y), interpolation=cv2.INTER_CUBIC)  


            if (self.do_normalize_data):
                print('Normalizing data...')
                print('Original data, minimum value:'+ str(np.min(helio_data)))
                print('Original data, maximum value:'+ str(np.max(helio_data)))
                helio_data=self.normalize_0_1(helio_data)                
                #print('Normalized data, minimum value:'+ str(np.min(helio_data)))
                #print('Normalized data, maximum value:'+ str(np.max(helio_data))) 

    
            self.data[file_i,0,:,:]=np.copy(helio_data)  

        print('Red '+str(file_i)+' files')
     
        return file_i


    
    def load_file_data(self, filename):
       
        print('Loading data from:'+filename,'...')

        self.data=np.empty([1,self.channels_n,self.size_x,self.size_y],dtype=float)
        self.data_frames_n = 1

        print('Reading data from:' +filename)            
            
        if (( filename.find('fits') != -1) or ( filename.find('fts') != -1) ):
            
            print('Loading fits data...')    
            try:    
                
                helio_data_fits = pyfits.open(path+'/'+files_arr[file_i])
                helio_data = np.copy(helio_data_fits[0].data)
    
            except:
                sys.exit("Failed to load FITS data ")
        
        else:
             
            print('Loading numpy data.... ')    
            try: 
                helio_data=np.load(path+'/'+files_arr[file_i])

            except:
                sys.exit("Unable to load numpy data ")
            

        if (self.do_normalize_data):
            print('Original data, maximum value:'+ str(np.max(helio_data)))
            print('Original data, maximum value:'+ str(np.max(helio_data)))
            helio_data=self.normalize_0_1(helio_data)                
            print('Normalized data, maximum value:'+ str(np.max(helio_data)))
            print('Normalized data, maximum value:'+ str(np.max(helio_data))) 

        helio_data_resize = np.resize(helio_data, (self.size_x,self.size_y)) 
        self.data[file_i,0,:,:]=np.copy(helio_data_resize)      
      
        return 1
    
    


    def pred_to_imgs_modes(self, pred, patch_height, patch_width, mode="sigma_theshold", sigma_threshold=1):
        
        """
        Convert predictions to image format.

        Parameters:
        pred (np.array): Predictions array of shape (Npatches, height*width, 2).
        patch_height (int): Height of each patch.
        patch_width (int): Width of each patch.
        mode (str): Mode of conversion ('original', 'threshold', or 'sigma_threshold').
        sigma_threshold (float): Number of standard deviations for 'sigma_threshold' mode.

        Returns:
        np.array: Converted predictions in image format.
        """
        # Threshold  'threshold' mode
        thrhold = (np.amax(pred) - np.amin(pred)) / 2

        # Mean and standard deviation for 'sigma_threshold' mode
        mean = np.mean(pred)
        std = np.std(pred)
    
        assert (len(pred.shape) == 3)  # Check for 3D array
        assert (pred.shape[2] == 2)  # Check for binary classification

        pred_images = np.zeros((pred.shape[0], pred.shape[1]))  # Initialize pred_images array

        if mode == "original":
            pred_images = pred[:, :, 1]

        elif mode == "threshold":
            pred_images = (pred[:, :, 1] >= thrhold).astype(int)

        elif mode == "sigma_threshold":
            sigma_thrhold = mean + sigma_threshold * std
            #sigma_thrhold = sigma_threshold * std
            pred_images = (pred[:, :, 1] >= sigma_threshold).astype(int)

        else:
            print("mode " + str(mode) + " not recognized. It can be 'original', 'threshold', or 'sigma_threshold'")
            exit() 

        pred_images = np.reshape(pred_images, (pred_images.shape[0], 1, patch_height, patch_width))
        return pred_images.astype(np.float)    
  


    def model_load_files(self, shape_fname, model_fname):
          
        try:
            self.model = tf.keras.models.model_from_json(open(shape_fname).read())
            self.model.load_weights(model_fname)         
            self.model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    
        except:
            sys.exit("Unable to load model files ")


    def model_predict(self):

        print('Running predictions:')
        predictions_1d = self.model.predict(self.data, verbose=self.verbose )
        #predictions_1d = self.normalize_0_1(predictions_1d)
        self.predictions= self.pred_to_imgs_modes(predictions_1d, self.size_x, self.size_y, self.output_mode, self.predict_sigma_threshold) 
        self.predictions[np.isnan(self.predictions)] = 0
    
        #self.predictions = np.transpose(self.predictions,(0,3,1,2))
        print("Predicted masks shape :")
        print(self.predictions.shape)
        print("Predicted data shape:")
        print(self.data.shape)
        print("Predicted data maximum value")
        print(np.amax(self.predictions))
        print("Predicted data minimum value")
        print(np.amin(self.predictions)) 

     
        for frame_i in range(0,self.data_frames_n):       
                   
            if (self.window_filt_predictions==True):
                print('Running window filter over predictions')
                self.predictions[frame_i,0,:,:]=np.copy(self.window_filt(self.predictions[frame_i,0,:,:],self.window_filt_boxsize, self.window_filt_increment, 0,self.size_x, self.size_y))               




    def save_predictions(self):
        
        print('Saving predictions...')
   
        for frame_i in range(0,self.data_frames_n):
        
            fname_predict_np = self.predict_path +'predicted_mask_'+str(frame_i)  
            fname_data = self.predict_path+'data_'+str(frame_i)
            fname_predict_img = self.predict_path +'predicted_mask_'+str(frame_i)+'.png' 
            fname_data_img = self.predict_path+'data_'+str(frame_i)+'.png'   
            fname_data_img_alt = self.predict_path+'data_alt_'+str(frame_i)+'.png'
            
            print('Saving initial and predicted data (numpy arrays)')
            np.save(fname_predict_np, self.predictions[frame_i], allow_pickle='false', fix_imports='false')
            np.save(fname_data, self.data[frame_i], allow_pickle='false', fix_imports='false' )          
            
            print('Saving initial and predicted data (png)')
   
            self.save_data_png_ri(self.predictions[frame_i,0],fname_predict_img);   
            self.save_data_png_ri(self.data[frame_i,0],fname_data_img);

            plt.imsave(arr = self.data[frame_i,0], fname= fname_data_img_alt, cmap='gray')
    


    def __init__(self, files_n, max_files_n):
        
       
        self.data_frames_n = min(files_n,max_files_n) 
     
        self.channels_n = 1 
    
        self.size_x = 1024
        self.size_y = 1024 #264

        self.data=np.empty([self.data_frames_n,self.channels_n,self.size_x,self.size_y],dtype=float)
        self.predictions=np.empty([self.data_frames_n,self.channels_n,self.size_x,self.size_y],dtype=float)
        
        self.do_normalize_data = True

        #Prediction parameters 
        self.predict_sigma_threshold = 0.6
        self.output_mode='original' # original or sigma_threshold
        self.window_filt_predictions = True
        self.window_filt_boxsize = 10
        self.window_filt_increment = 1

        self.verbose = 2

        self.shape_name = 'model_shape'
        self.model_name = 'model_weights'
        self.predict_path = './predictions/'
        