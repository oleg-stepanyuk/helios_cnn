import sys
import os
import numpy as np
import cv2
from skimage import exposure
from skimage.util import img_as_uint
from astropy.io import fits

# NEW: ONNXRuntime import
import onnxruntime as ort


# Function to save data as PNG (no compression)
def save_data_png(data, fname):
    img_arr = exposure.rescale_intensity(data, out_range='float')
    img_arr = img_as_uint(img_arr)
    cv2.imwrite(fname, img_arr, [cv2.IMWRITE_PNG_COMPRESSION, 0])


# Normalize data to 0-1 range
def normalize_0_1(data_arr):
    data_arr_no_nan = np.nan_to_num(data_arr)
    data_arr_min = np.amin(data_arr_no_nan)
    data_arr_from_zero = np.subtract(data_arr_no_nan, data_arr_min)
    data_arr_max = np.amax(data_arr_from_zero)
    if data_arr_max != 0:
        return np.multiply(data_arr_from_zero, 1 / data_arr_max)
    else:
        return np.zeros_like(data_arr)


def load_fits_data_no_wcs(path, hdu=0, plane=0):
    """
    Load FITS/FTS image data WITHOUT requiring WCS metadata.
    Returns a 2D float32 array.
    """
    with fits.open(path, memmap=False) as hdul:
        data = hdul[hdu].data

    if data is None:
        raise ValueError(f"No image data found in HDU {hdu} for {path}")

    arr = np.asarray(data)
    arr = np.squeeze(arr)

    # If 3D or more, take a plane along axis 0
    if arr.ndim == 2:
        img2d = arr
    elif arr.ndim >= 3:
        if not (0 <= plane < arr.shape[0]):
            raise ValueError(f"plane {plane} out of range for data shape {arr.shape}")
        img2d = np.squeeze(arr[plane])
        if img2d.ndim != 2:
            raise ValueError(f"Selected plane is not 2D: got {img2d.shape}")
    else:
        raise ValueError(f"Unsupported FITS data dims: {arr.ndim} (shape {arr.shape})")

    img2d = np.nan_to_num(img2d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return img2d


def _ensure_2d_grayscale(arr: np.ndarray, source_name: str = "input") -> np.ndarray:
    """
    Make common stored forms usable as a 2D grayscale image:
      - (H, W)                    -> OK
      - (1, H, W)                 -> squeeze -> (H, W)
      - (H, W, 1)                 -> squeeze -> (H, W)
      - (C, H, W) where C in 1/3/4 -> take first channel -> (H, W)
      - (H, W, C) where C in 1/3/4 -> take first channel -> (H, W)
    Otherwise error out.
    """
    arr = np.asarray(arr)

    # Drop singleton dimensions (fixes your case: (1,264,264) -> (264,264))
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        # (C, H, W)
        if arr.shape[0] in (1, 3, 4) and arr.shape[1] > 1 and arr.shape[2] > 1:
            return arr[0, :, :]
        # (H, W, C)
        if arr.shape[2] in (1, 3, 4) and arr.shape[0] > 1 and arr.shape[1] > 1:
            return arr[:, :, 0]

    raise ValueError(f"{source_name} must be a 2D grayscale image; got shape {arr.shape}")


# Load data from file (FITS, PNG, or NumPy)
def load_file_data(filename, do_normalize=True, resize_to=None):
    """
    If resize_to is None: keep original size (size-agnostic default).
    If resize_to is (W,H): resize to that size.
    Returns data shaped [1, H, W, 1] float32.
    """
    print(f'Loading data from: {filename}')

    if filename.endswith('.fits') or filename.endswith('.fts'):
        print('Loading FITS data...')
        try:
            print("Loading FITS data (astropy, no WCS required)...")
            helio_data = load_fits_data_no_wcs(filename, hdu=0, plane=0)
        except Exception as e:
            print(f'Failed to load FITS data: {e}')
            sys.exit(1)

    elif filename.endswith('.png'):
        print('Loading PNG data...')
        helio_data = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if helio_data is None:
            print('Failed to load PNG data.')
            sys.exit(1)

    elif filename.endswith('.npy'):
        print('Loading NumPy data...')
        try:
            helio_data = np.load(filename)
        except Exception as e:
            print(f'Failed to load NumPy data: {e}')
            sys.exit(1)

    else:
        print('Unsupported file type. Supported: .fits, .fts, .png, .npy')
        sys.exit(1)

    print("helio_data dtype:", getattr(helio_data, "dtype", type(helio_data)))
    print("helio_data shape:", getattr(helio_data, "shape", None))
    try:
        print("helio_data min/max:", float(np.nanmin(helio_data)), float(np.nanmax(helio_data)))
    except Exception:
        pass

    # ✅ FIX: accept (1,H,W), (H,W,1), etc.
    try:
        helio_data = _ensure_2d_grayscale(helio_data, source_name=filename)
    except Exception as e:
        print(str(e))
        sys.exit(1)

    # Rescale intensity, cast to uint16 (same behavior you had)
    helio_data = exposure.rescale_intensity(helio_data, out_range='float')
    helio_data = img_as_uint(helio_data)

    # Optional resize
    if resize_to is not None:
        # resize_to is (W,H) because cv2 wants (W,H)
        helio_data = cv2.resize(helio_data, dsize=resize_to, interpolation=cv2.INTER_CUBIC)

    if do_normalize:
        print('Normalizing data...')
        helio_data = normalize_0_1(helio_data)

    # Prepare for model: [1, height, width, 1] (channels_last)
    data = np.expand_dims(helio_data, axis=0)   # -> [1,H,W]
    data = np.expand_dims(data, axis=-1)        # -> [1,H,W,1]
    return data.astype(np.float32)


# Convert predictions to image format
def pred_to_imgs_modes(pred, height, width, mode='original', sigma_threshold=0.6):
    assert len(pred.shape) == 3 and pred.shape[2] == 2
    pred_images = np.zeros((pred.shape[0], pred.shape[1]))

    if mode == 'original':
        pred_images = pred[:, :, 1]
    elif mode == 'threshold':
        threshold = (np.amax(pred) - np.amin(pred)) / 2
        pred_images = (pred[:, :, 1] >= threshold).astype(int)
    elif mode == 'sigma_threshold':
        mean = np.mean(pred)
        std = np.std(pred)
        sigma_thr = mean + sigma_threshold * std
        pred_images = (pred[:, :, 1] >= sigma_thr).astype(float)
    else:
        print(f'Unknown mode: {mode}. Use "original", "threshold", or "sigma_threshold"')
        sys.exit(1)

    pred_images = np.reshape(pred_images, (pred_images.shape[0], 1, height, width))
    return pred_images


# -------- ONNX helper functions --------
def _onnx_pick_input_layout(sess: ort.InferenceSession) -> str:
    """
    Decide NCHW vs NHWC from the ONNX model input shape.
    Robust: looks for channel dimension == 1 (or 3) in position 1 or 3.
    Falls back to NHWC because your preprocessing naturally produces NHWC.
    """
    inp = sess.get_inputs()[0]
    shp = inp.shape  # can contain None / strings
    if not isinstance(shp, (list, tuple)) or len(shp) != 4:
        return "NHWC"

    def _as_int(x):
        try:
            return int(x)
        except Exception:
            return None

    d1 = _as_int(shp[1])
    d3 = _as_int(shp[3])

    if d1 in (1, 3, 4):
        return "NCHW"
    if d3 in (1, 3, 4):
        return "NHWC"
    return "NHWC"


def _onnx_prepare_input(data_nhwc: np.ndarray, layout: str) -> np.ndarray:
    """
    data_nhwc is [1,H,W,1] as produced by load_file_data.
    Convert to required layout for the ONNX model.
    """
    if layout == "NHWC":
        return data_nhwc.astype(np.float32)
    return np.transpose(data_nhwc, (0, 3, 1, 2)).astype(np.float32)


def _onnx_fix_output_to_hw2(y: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Ensure output ends up as shape [1, H*W, 2] to satisfy pred_to_imgs_modes.
    Accepts common variants:
      [1, H*W, 2]         -> OK
      [1, 2, H*W]         -> transpose
      [H*W, 2]            -> add batch
      [2, H*W]            -> transpose + add batch
      [1, H, W, 2]        -> reshape to [1, H*W, 2]
      [1, 2, H, W]        -> transpose then reshape
    """
    y = np.asarray(y)

    if y.ndim == 2:
        if y.shape[1] == 2 and y.shape[0] == h * w:
            return y[None, :, :].astype(np.float32)
        if y.shape[0] == 2 and y.shape[1] == h * w:
            return np.transpose(y, (1, 0))[None, :, :].astype(np.float32)

    if y.ndim == 3:
        if y.shape[0] == 1 and y.shape[2] == 2 and y.shape[1] == h * w:
            return y.astype(np.float32)
        if y.shape[0] == 1 and y.shape[1] == 2 and y.shape[2] == h * w:
            return np.transpose(y, (0, 2, 1)).astype(np.float32)

    if y.ndim == 4:
        if y.shape[0] == 1 and y.shape[1] == h and y.shape[2] == w and y.shape[3] == 2:
            return y.reshape(1, h * w, 2).astype(np.float32)
        if y.shape[0] == 1 and y.shape[1] == 2 and y.shape[2] == h and y.shape[3] == w:
            y_hwc2 = np.transpose(y, (0, 2, 3, 1))
            return y_hwc2.reshape(1, h * w, 2).astype(np.float32)

    raise ValueError(
        f"Unsupported ONNX output shape {y.shape}; can't convert to [1, H*W, 2] with H={h}, W={w}."
    )


def main():
    if len(sys.argv) < 4:
        print('Usage: onnx segment.py <model.onnx> <unused_h5> <input_file> <output_prefix> '
              '[--manual_resize <height> <width>] [--threshold <sigma_threshold>] [--no_normalize]')
        sys.exit(1)

    model_onnx = sys.argv[1]
    input_file = sys.argv[2]
    output_prefix = sys.argv[3]

    manual_resize_idx = sys.argv.index('--manual_resize') if '--manual_resize' in sys.argv else -1
    manual_h = None
    manual_w = None
    if manual_resize_idx != -1:
        manual_h = int(sys.argv[manual_resize_idx + 1])
        manual_w = int(sys.argv[manual_resize_idx + 2])

    sigma_threshold = float(sys.argv[sys.argv.index('--threshold') + 1]) if '--threshold' in sys.argv else 0.6
    do_normalize = '--no_normalize' not in sys.argv

    if manual_h is not None and manual_w is not None:
        if manual_h % 4 != 0 or manual_w % 4 != 0:
            print('Warning: Input size should be divisible by 4 for proper up/down sampling alignment.')
        resize_to = (manual_w, manual_h)  # cv2 expects (W,H)
        print(f'Resizing input to: {manual_h} x {manual_w}')
    else:
        resize_to = None
        print('No resize requested: using original input size.')

    data_nhwc = load_file_data(input_file, do_normalize=do_normalize, resize_to=resize_to)

    h = int(data_nhwc.shape[1])
    w = int(data_nhwc.shape[2])
    print(f'Actual model input tensor size: {h} x {w}')

    print('Loading ONNX model...')
    try:
        sess = ort.InferenceSession(model_onnx, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f'Failed to load ONNX model from {model_onnx}: {e}')
        sys.exit(1)

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    layout = _onnx_pick_input_layout(sess)
    model_input = _onnx_prepare_input(data_nhwc, layout)

    print(f'Running prediction with ONNXRuntime... (input layout: {layout})')
    try:
        y = sess.run([output_name], {input_name: model_input})[0]
    except Exception as e:
        print(f'ONNX inference failed: {e}')
        sys.exit(1)

    try:
        predictions_1d = _onnx_fix_output_to_hw2(y, h, w)
    except Exception as e:
        print(f'Failed to adapt ONNX output shape: {e}')
        print(f'Raw output shape was: {np.asarray(y).shape}')
        sys.exit(1)

    predictions = pred_to_imgs_modes(
        predictions_1d, h, w, mode='sigma_threshold', sigma_threshold=sigma_threshold
    )
    predictions[np.isnan(predictions)] = 0

    print('Saving predictions...')
    if os.path.dirname(output_prefix):
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    fname_predict_np = output_prefix + '_mask.npy'
    fname_predict_img = output_prefix + '_mask.png'
    fname_data_np = output_prefix + '_data.npy'
    fname_data_img = output_prefix + '_data.png'

    np.save(fname_predict_np, predictions[0], allow_pickle=False)
    np.save(fname_data_np, data_nhwc[0], allow_pickle=False)
    save_data_png(predictions[0, 0, :, :], fname_predict_img)
    save_data_png(data_nhwc[0, :, :, 0], fname_data_img)

    print('Done.')


if __name__ == '__main__':
    main()
