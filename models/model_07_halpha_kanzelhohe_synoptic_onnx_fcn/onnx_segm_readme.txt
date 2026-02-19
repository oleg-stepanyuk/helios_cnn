# 🌞 Solar Eruptive Phenomena Segmentation (ONNX)

Image segmentation of eruptive solar phenomena using a pre-trained ONNX
model and ONNX Runtime.

Main script: `onnx_segm.py`

------------------------------------------------------------------------

## ✨ Features

-   FITS / FTS, PNG, and NumPy (.npy) 
-   Sigma-based adaptive thresholding\
-   Saves results as PNG and NumPy

------------------------------------------------------------------------

## 🔧 Installation

``` bash
pip install numpy opencv-python scikit-image astropy onnxruntime
```

------------------------------------------------------------------------

## 🚀 Usage

``` bash
python onnx_segm.py <model.onnx> <input_file> <output_prefix>     [--manual_resize <height> <width>]     [--threshold <sigma_threshold>]     [--no_normalize]
```

### Required Arguments

-   `model.onnx` -- trained ONNX segmentation model\
-   `input_file` -- `.fits`, `.fts`, `.png`, or `.npy`\
-   `output_prefix` -- prefix for output files

------------------------------------------------------------------------

## 📦 Output Files

    <output_prefix>_mask.npy
    <output_prefix>_mask.png
    <output_prefix>_data.npy
    <output_prefix>_data.png

------------------------------------------------------------------------

## 🧠 Model Requirements

-   Single-channel input\
-   2-channel output (background / eruptive feature)

Compatible output shapes include:

    [1, H, W, 2]
    [1, 2, H, W]
    [1, H*W, 2]

------------------------------------------------------------------------

## Applications

-   Solar physics research\
-   Space weather monitoring\
-   Automated flare detection
