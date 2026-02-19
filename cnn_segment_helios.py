from cnn_predictor import t_cnn_predictor  # Only keep if needed by t_cnn_predictor
import sys
import os
from tensorflow.keras import backend as K


def main():
    
    K.clear_session() 
    
    print('Image segmantation routine for CNNs trained on solar eruptive phenomena data. Written by Oleg Stepanyuk. \n')
  
    size_x = 264
    size_y = 264 
    predict_sigma_threshold = 0
    max_files_n = 0 

    if  ((len(sys.argv)) == 6): 
        print('WARNING:')
        print('1) It is recommended to specify threshold (STD Sigma units) to get proper feature masks. For now moving on with "0"')
        print('The segmentation routine output can be converted into binary masks through thresholding (STD deviation units.')
        print('By choosing a probability threshold, pixels with probabilities above this threshold are classified as part of a feature')
        print('while those below are not. This threshold can be adjusted based on specific requirements or desired sensitivity')  
        print('')
        print('2) It is advised to specify resize parameters for the images (size_x*sizy_y) so that they correspond to the size of')
        print('Images the model was trained on')
        predict_sigma_threshold = 0
    

    elif ( (len(sys.argv) == 7) or (len(sys.argv) == 9)):
        predict_sigma_threshold = float(sys.argv[6])
                    
        if ((len(sys.argv)) ==9):
            size_x = int(sys.argv[7])
            size_y = int(sys.argv[8])
        
        else: 
            print('1) It is advised to specify resize parameters for the images (size_x*sizy_y) so that they correspond to the size of')
            print('Images the model was trained on')
            print('Moving on with size_x=264, size_y=264')
         
    else: 
    
        print('Syntax:')                                   
        print('> python3 wavetrack_unet.py <model_shape.json> <model_weights.h5> <-f/-s> <input_data_path / input_data_filename> <predictions_path> <sigma_threshold> <resize_x> <resize_y>'  )
        sys.exit() 
         
    if (sys.argv[3]=='-s'):
        print('Processing single file...') 
        process_folder=False
    
    elif (sys.argv[3]=='-f'): 
        print('Multiple file(folder) mode...')   
        process_folder=True

    else:
        print('Syntax:')      #                  1                  2                 3            4                                       5                  6             7          8                                 
        print('> python3 wavetrack_unet.py <model_shape.json> <model_weights.h5>  <-f/-s> <input_data_path / input_data_filename> <predictions_path> <sigma_threshold> <resize_x> <resize_y>'  )
        print('You need to specify if you want to perform images segmentation of a single file "-s", or the whole folder "-f"')
        sys.exit()
    
    if (process_folder):

        try:
       
            data_files_arr = os.listdir(sys.argv[4])
            data_files_n = len(data_files_arr)
            
        except: 
        
            print('Error: unable to read data from the folder.')  
            sys.exit() 
    
    else: 
            data_files_n = 1   
    

    ml_seg = t_cnn_predictor(data_files_n,max_files_n)
    ml_seg.size_x = size_x
    ml_seg.size_y = size_y 
    ml_seg.predict_sigma_threshold = predict_sigma_threshold 
    ml_seg.predict_path = sys.argv[5]
    ml_seg.model_load_files(sys.argv[1],sys.argv[2])
    
    ml_seg.max_files_n = 100
        

    if (process_folder):
        ml_seg.load_folder_data(sys.argv[4])
    
    else: 
        ml_seg.load_file_data(seg.argv[4])

 

    ml_seg.model_predict() 
    ml_seg.save_predictions()
            



if __name__ == "__main__":
    main()




