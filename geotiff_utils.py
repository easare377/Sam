from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr

import numpy as np
import pandas as pd
import image_utils as iu
import tensorflow as tf
import os
from PIL import Image, ImageEnhance
from PIL import ImageFilter
from keras.models import load_model
from shutil import copy2
import matplotlib.pyplot as plt


def get_rgb_bands(tiff, color_depth='uint8'):
    red = tiff.GetRasterBand(1).ReadAsArray()
    green = tiff.GetRasterBand(2).ReadAsArray()
    blue = tiff.GetRasterBand(3).ReadAsArray()
    #rgbOutput = source.ReadAsArray() #Easier method
    rgbOutput = np.zeros((tiff.RasterYSize, tiff.RasterXSize, 3), color_depth)
    rgbOutput[..., 0] = red
    rgbOutput[..., 1] = green
    rgbOutput[..., 2] = blue
    #Clear so file isn't locked
    source = None
    return rgbOutput


def get_l_band(tiff, band, color_depth='uint8'):
    L = tiff.GetRasterBand(band).ReadAsArray()
    #rgbOutput = source.ReadAsArray() #Easier method
    rgbOutput = np.zeros((tiff.RasterYSize, tiff.RasterXSize, 1), color_depth)
    rgbOutput[..., 0] = L
    #Clear so file isn't locked
    source = None
    return rgbOutput


def save_image_as_geotiff(mask, destination_path, dataset):
    mask = mask.resize((dataset.RasterXSize, dataset.RasterYSize))
    array = np.asarray(mask)
    nrows, ncols = array.shape[0], array.shape[1]
    depth = array.shape[2]
    # geotransform = (0, 1024, 0, 0, 0, 512)
    # That's (top left x, w-e pixel resolution, rotation (0 if North is up),
    #         top left y, rotation (0 if North is up), n-s pixel resolution)
    # I don't know why rotation is in twice???
    output_raster = gdal.GetDriverByName('GTiff').Create(
        destination_path, ncols, nrows, depth, gdal.GDT_Byte)  # Open the file
    # print(output_raster)
    output_raster.SetGeoTransform(
        dataset.GetGeoTransform())  # Specify its coordinates
    output_raster.SetProjection(dataset.GetProjection())
    output_raster.SetMetadata(output_raster.GetMetadata())
    # srs = osr.SpatialReference()                 # Establish its coordinate encoding
    # This one specifies WGS84 lat long.
#     if unit == 'meters':
#         srs.ImportFromEPSG(3857)
#     elif unit == 'degrees':
#         srs.ImportFromEPSG(4326)
#    # Anyone know how to specify the
#    # IAU2000:49900 Mars encoding?
#    # Exports the coordinate system
#     output_raster.SetProjection(srs.ExportToWkt())
   # to the file
    for x in range(depth):
       output_raster.GetRasterBand(x + 1).WriteArray(
           array[:, :, x])   # Writes my array to the raster
    output_raster.FlushCache()


def read_geoTiff_bands(geoTiff, dtype='uint16'):
    width = geoTiff.RasterXSize
    height = geoTiff.RasterYSize
    bands = []
    for x in range(geoTiff.RasterCount):
        band = geoTiff.GetRasterBand(x + 1).ReadAsArray()
        bands.append(band)
    output = np.zeros(
        (int(height), int(width), geoTiff.RasterCount), dtype)
    for x in range(len(bands)):
        output[..., x] = bands[x]
    return output
