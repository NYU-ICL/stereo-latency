from pathlib import Path

import numpy as np

import OpenEXR as exr
import Imath

def imread(path: Path):
    exrfile = exr.InputFile(path.as_posix())
    header = exrfile.header()

    dataWindow = header["dataWindow"]
    imsize = (dataWindow.max.y - dataWindow.min.y + 1, dataWindow.max.x - dataWindow.min.x + 1, 1)

    channel_data_dict = {}
    for channel in header["channels"]:
        channel_data = exrfile.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
        channel_data = np.fromstring(channel_data, dtype=np.float32)
        channel_data = channel_data.reshape(imsize)
        channel_data_dict[channel] = channel_data

    color_channels = ["R", "G", "B"]
    img = np.concatenate([channel_data_dict[c] for c in color_channels], axis=-1)
    img = gamma_encode(img)
    img = np.clip(img, 0, 1)

    depth_channel = "Z"
    depth = None if depth_channel not in header["channels"] else channel_data_dict[depth_channel]

    return img, depth


def gamma_encode(img):
    img = np.where(img <= 0.0031308, 12.92 * img, 1.055 * np.power(img, 1/2.4) - 0.055)
    return img
