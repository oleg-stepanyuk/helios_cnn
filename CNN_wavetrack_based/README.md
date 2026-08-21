

The script  cnn_segment_helios.py is designed for segmenting images using a CNN model trained on solar eruptive phenomena. Before usage, ensure that you have Python and necessary libraries installed (like TensorFlow, Keras, NumPy, and OpenCV).


1 Prerequisites

	1.1 Python: Ensure Python 3.x is installed.

	1.2 Dependencies: Install required libraries if not already installed:
	> pip install tensorflow keras numpy opencv-python matplotlib scikit-image astropy sunpy aiapy



2. Files Required by the script
	
	model_shape.json: Model architecture file.
	model_weights.h5: Trained model weights.
	input_data: Directory path or filename for input images.
	predictions_path: Directory path to save output predictions

	Each of the models is usually accompanied by it's own description. 


3. Running the Script
	

	The script can be run from the terminal or command prompt. 

    Add on single/folder file usage. ..

	3.1 Basic Usage
    	To run the script with basic settings (default thresholds and image sizes), use:
		>python cnn_segment_helios.py model_shape.json model_weights.h5 input_data_path predictions_path

    3.2 Probabalistic output (certain models) and threshold. 

        Outputting a range of values, certain models provide a more detailed and flexible interpretation of the image. A probabilistic output allows for a more nuanced interpretation of ambiguous region. The continuous output can be converted into binary masks through thresholding. By choosing a probability threshold, pixels with probabilities above this threshold are classified as part of a feature, while those below are not. This threshold can be adjusted based on specific requirements or desired sensitivity, providing flexibility in how the results are interpreted and used. The range of values also gives insights into the model's confidence. 

	
	3.2 Advanced Usage
    	To specify custom options like image size and threshold:

		With a Sigma Threshold and Default Image Size:
		>python cnn_segment_helios.py model_shape.json model_weights.h5 input_data_path predictions_path sigma_threshold
    
    	With Both Sigma Threshold and Image Size:
    	python cnn_segment_helios.py model_shape.json model_weights.h5 input_data_path predictions_path sigma_threshold size_x size_y




    3.3 Parameters Explanation

		model_shape.json: Path to the JSON file containing the CNN model architecture.
		model_weights.h5: Path to the H5 file containing the weights of the pretrained model.
		input_data_path: Path to the input data (can be a directory or a specific file).
		predictions_path: Path where the segmented outputs will be saved.
		sigma_threshold (optional): The threshold for feature detection, specified in standard deviation units.
		size_x and size_y (optional): Dimensions to which the input images are resized.


    3.4 Example Command

    	>python cnn_segment_helios.py architecture.json weights.h5 /data/solar_images /output/predictions 0.5 256 256

    	This command runs the segmentation on images located at /data/solar_images using a CNN model described in architecture.json with weights from weights.h5, a sigma threshold of 0.5, and resizes images to 256x256 pixels before processing. The results are saved in /output/predictions.


4. The script cnn_predictor.py used by the main script cnn_segment_helios.py defines a Python class named t_cnn_predictor which handles processing and prediction tasks for images

    

	4.1 Image Saving and Conversion:

	The method save_data_png_ri is used to save image data as PNG files. It involves rescaling the image intensity to float range, converting the image array to an unsigned integer format, and writing it to a file without compression.

	4.2 Image Processing:

	The method window_filt appliess a window filtering technique to the image data. It processes the image in small sections (or windows), iterating through the image based on specified box size and increments. This method is designed to filter smaller features in the image data, in case predicted binary masks contain too much small spurious features



