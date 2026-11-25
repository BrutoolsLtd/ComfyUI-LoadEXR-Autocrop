# ComfyUI Load EXR (Auto Crop)

A ComfyUI custom node for loading a single EXR image or a numbered EXR sequence. It outputs an RGB image batch and an alpha mask.

## Features

- Loads a single `.exr` file or a sequence using a `%04d` frame pattern.
- Optionally crops the EXR Data Window to the Display Window.
- Supports linear, sRGB, and Reinhard RGB output.
- Returns the EXR alpha channel as a ComfyUI mask when available.

## Installation

1. Place this custom node folder in `ComfyUI/custom_nodes/`.
2. Make sure `OpenImageIO` is installed in the same Python environment used to run ComfyUI.
3. Restart ComfyUI.

ComfyUI provides PyTorch. The node also imports NumPy, OpenCV (`cv2`), and `tqdm`. OpenCV must have OpenEXR support enabled.

## Usage

Add **Load EXR (Auto Crop to Display)** to the workflow.

### Inputs

- **filepath** — Path to a single `.exr` file or a frame sequence pattern containing `%04d`, such as `renders/frame.%04d.exr`. If the exact path exists, it is loaded as a single file.
- **tonemap**
  - `linear`: leaves RGB values linear and unclamped.
  - `sRGB`: applies the linear-to-sRGB transfer function and clamps RGB to `[0, 1]`.
  - `Reinhard`: clamps negative RGB values, applies `x / (x + 1)`, converts to sRGB, and clamps to `[0, 1]`.
- **start_frame** — Inclusive first frame for a `%04d` sequence.
- **end_frame** — Inclusive last frame for a `%04d` sequence.
- **auto_crop**
  - `True`: reads with OpenImageIO and crops to the Display Window when the Data Window and Display Window differ.
  - `False`: reads with OpenCV and does not crop to the Display Window.

For example, with `filepath` set to `renders/frame.%04d.exr`, `start_frame` set to `1001`, and `end_frame` set to `1025`, the node loads frames 1001 through 1025.

### Outputs

- **RGB** — ComfyUI `IMAGE`, shaped `[batch, height, width, 3]`.
- **alpha** — ComfyUI `MASK`, shaped `[batch, height, width]`. The EXR alpha channel is copied as-is and clamped to `[0, 1]`; it is not inverted. If the OpenCV path reads an image without alpha, the mask is zero-filled.
- **batch_size** — Number of loaded frames.
- **start_frame** — The supplied start-frame value.

## EXR channel requirements

When `auto_crop=True`, OpenImageIO channel names must be `R`, `G`, `B`, and `A`, in that order. The check is case-insensitive. Files with another channel layout raise an error instead of being silently reordered.

The OpenImageIO path converts RGB(A) channel order to BGR(A) internally so the common RGB conversion produces the expected output. The OpenCV path uses OpenCV's channel ordering directly.

## Notes

- The Display Window crop is performed only when the Data Window and Display Window differ.
- OpenCV's `OPENCV_IO_ENABLE_OPENEXR` setting is enabled by the node when it is not already set. This cannot provide EXR support if the installed OpenCV build lacks the codec.