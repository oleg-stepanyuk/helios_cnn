import sys
import argparse
#from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import plot_model
import tensorflow as tf
#import pudot

def load_model(json_path, model_path):
    # Load the model structure from the JSON file
    with open(json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    #model = model_from_json(loaded_model_json)
    model = tf.keras.models.model_from_json(loaded_model_json)
    # Load weights into the model
    model.load_weights(model_path)
    
    return model

def plot_model_architecture(model, output_file='model_plot.png'):
    # Generate the plot
    plot_model(model, to_file=output_file, show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=True)
    print(f"Model plot saved to {output_file}")



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot Keras model architecture.')
    parser.add_argument('json_path', type=str, help='Path to the model JSON file.')
    parser.add_argument('model_path', type=str, help='Path to the model weights file.')
    args = parser.parse_args()

    # Load and plot the model
    model = load_model(args.json_path, args.model_path)
    plot_model_architecture(model)

if __name__ == '__main__':
    main()
