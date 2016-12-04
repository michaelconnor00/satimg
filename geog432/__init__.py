"""
Helper Functions for GEOG432 Project.
"""
import os
from osgeo import gdal, gdal_array
from matplotlib import pyplot
import numpy as np
import rsgislib
from rsgislib import imageutils

gdal.UseExceptions()

# def stack_bands(path):
#     """
#     This function will read all files with a TIF, TIFF, tiff, tif extension
#     read as an array combine and write out as an array.
#     NOTE: order of the files matters, makes sure the files are in order so they are stacked correctly.
#     :param path: path to the images to be stacked.
#     """
#     files = os.listdir(path)
#     base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#     band_arrays = []
#     for band in files:
#         extension = os.path.splitext(band)[1]
#         if extension in ['.TIF', '.TIFF', '.tiff', '.tif']:
#             full_filename = os.path.join(base_dir, path, band)
#             print('Reading: %s' % full_filename)
#             img = gdal.Open(full_filename)
#             band_ary = img.ReadAsArray()
#             band_arrays.append(band_ary)
#     stacked = np.array(band_arrays)
#     # print(type(stacked.astype("int")), stacked.astype("int"))
#     output_file = os.path.join(base_dir, path, "stacked.tif")
#     proto = os.path.join(base_dir, path, files[0])
#     out_array = gdal_array.SaveArray(stacked, output_file, "GTiff", gdal.Open(proto))
#     out_array = None
#     print('Stacked GTIff written to %s' % output_file)

def stack_bands(path):
    """
    This function will read all files with a TIF, TIFF, tiff, tif extension
    read as an array combine and write out as an array.
    NOTE: order of the files matters, makes sure the files are in order so they are stacked correctly.
    :param path: path to the images to be stacked.
    """
    tif_exts = ['.TIF', '.TIFF', '.tiff', '.tif']
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    files = os.listdir(path)
    tif_files = [os.path.join(base_dir, f) for f in files if os.path.splitext(f)[1] in tif_exts]

    bandNamesList = ['Coastal','Blue','Green','Red','NIR','SWIR1','SWIR2']

    output_file = os.path.join(base_dir, path, "stacked.tif")

    # Set format and type
    gdalFormat = 'GTiff'
    dataType = rsgislib.TYPE_16UINT

    imageutils.stackImageBands(tif_files, None, output_file, None, 0, gdalFormat, dataType)


def img_to_band_data(img_path):
    """
    Take the *.vrt and build and plot.

    :param img_path: str path to the image to open
    :return: numpy array with band data
    """
    raster_dataset = gdal.Open(img_path, gdal.GA_ReadOnly)
    bands_data = []
    for b in range(1, raster_dataset.RasterCount+1):
        band = raster_dataset.GetRasterBand(b)
        bands_data.append(band.ReadAsArray())

    return np.dstack(bands_data)


def raster_show(bands_data, rgb_index):
    """
    Take the band data and displays the bands listed in rgb_index.

    :param bands_data: numpy array with band data
    :param rgb_index: a list (len=3) describing bands to display.
    """
    f = pyplot.figure()
    f.add_subplot(1, 1, 1)
    r = bands_data[:,:,rgb_index[0]]
    g = bands_data[:,:,rgb_index[1]]
    b = bands_data[:,:,rgb_index[2]]
    rgb = np.dstack([r,g,b])
    pyplot.imshow(rgb/255)
    pyplot.show()
