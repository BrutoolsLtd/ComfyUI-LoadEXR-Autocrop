import os
from tqdm import tqdm




import cv2 as cv
import torch
import numpy as np

import OpenImageIO as oiio

from comfy.utils import PROGRESS_BAR_ENABLED, ProgressBar


def linearToSRGB(npArray):
    less = npArray <= 0.0031308
    npArray[less] = npArray[less] * 12.92
    npArray[~less] = np.power(npArray[~less], 1/2.4) * 1.055 - 0.055

def load_exr_pixels(filepath):
    buf = oiio.ImageBuf(filepath)
    if buf.nchannels == 0:
        raise ValueError(f"No se pudo cargar el EXR: {filepath}")

    spec = buf.spec()
    data_window = (spec.x, spec.y, spec.width, spec.height)
    display_window = (
        spec.full_x,
        spec.full_y,
        spec.full_width,
        spec.full_height,
    )

    if data_window != display_window:
        buf = oiio.ImageBufAlgo.crop(buf, spec.display_window)

    return buf.get_pixels().astype(np.float32)

def load_EXR(filepath, tonemap, auto_crop):
    if auto_crop:
        image = load_exr_pixels(filepath)
    else:
        image = cv.imread(filepath, cv.IMREAD_UNCHANGED).astype(np.float32)

    if len(image.shape) == 2:
        image = np.repeat(image[..., np.newaxis], 3, axis=2)
    rgb = np.flip(image[:,:,:3], 2).copy()

    if tonemap == "sRGB":
        linearToSRGB(rgb)
        rgb = np.clip(rgb, 0, 1)
    elif tonemap == "Reinhard":
        rgb = np.clip(rgb, 0, None)
        rgb = rgb / (rgb + 1)
        linearToSRGB(rgb)
        rgb = np.clip(rgb, 0, 1)

    rgb = torch.unsqueeze(torch.from_numpy(rgb), 0)

    mask = torch.zeros((1, image.shape[0], image.shape[1]), dtype=torch.float32)
    if image.shape[2] > 3:
        mask[0] = torch.from_numpy(np.clip(image[:,:,3], 0, 1))

    return (rgb, mask)


class LoadEXRAutoCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filepath": ("STRING", {"default": "path/to/frame%04d.exr"}),
                "tonemap": (["linear", "sRGB", "Reinhard"], {"default": "sRGB"}),
                "start_frame": ("INT", {"default": 1001, "min": 0, "max": 9999}),
                "end_frame": ("INT", {"default": 1001, "min": 0, "max": 9999}),
                "auto_crop": ("BOOLEAN", {"default": True, "label_on": "True (Crop to DisplayWindow)", "label_off": "False (Use DataWindow)"}),
            },
        }

    CATEGORY = "folks/HQ-Image-Save"

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("RGB", "alpha", "batch_size", "start_frame")
    FUNCTION = "load"

    def load(self, filepath, tonemap, start_frame, end_frame, auto_crop):
        if os.path.splitext(os.path.normpath(filepath))[1].lower() != ".exr":
            raise Exception("Filepath needs to end in .exr or .EXR")

        frames = list(range(start_frame, end_frame+1))
        if len(frames) == 0:
            raise Exception("Invalid frame range")

        if os.path.exists(os.path.normpath(filepath)): # absolute mode
            rgb, mask = load_EXR(os.path.normpath(filepath), tonemap, auto_crop)
            batch_size = 1
        elif "%04d" in filepath: # frame substitution
            rgb = []
            mask = []
            batch_size = len(frames)

            if PROGRESS_BAR_ENABLED and batch_size > 1:
                pbar = ProgressBar(batch_size)
            else:
                pbar = None

            for frame in tqdm(frames, desc="loading images"):
                framepath = os.path.normpath(filepath.replace("%04d", f"{frame:04}"))
                if os.path.exists(framepath):
                    rgbFrame, maskFrame = load_EXR(framepath, tonemap)
                    rgb.append(rgbFrame)
                    mask.append(maskFrame)
                else:
                    raise Exception("Frame not found: " + framepath)
                if pbar is not None:
                    pbar.update(1)
            rgb = torch.cat(rgb, 0)
            mask = torch.cat(mask, 0)
        else:
            raise Exception("Path not found: " + filepath)

        return (rgb, mask, batch_size, start_frame)
