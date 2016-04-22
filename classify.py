"""
Classify a multi-band, satellite image.

Usage:
    classify.py <input_fname> <train_data_path> <output_fname> [--method=<classification_method>]
                                                               [--validation=<validation_data_path>]
                                                               [--verbose]
    classify.py -h | --help

The <input_fname> argument must be the path to a GeoTIFF image.

The <train_data_path> argument must be a path to a directory with vector data files
(in shapefile format). These vectors must specify the target class of the training pixels. One file
per class. The base filename (without extension) is taken as class name.

If a <validation_data_path> is given, then the validation vector files must correspond by name with
the training data. That is, if there is a training file train_data_path/A.shp then the corresponding
validation_data_path/A.shp is expected.

The <output_fname> argument must be a filename where the classification will be saved (GeoTIFF format).

No geographic transformation is performed on the data. The raster and vector data geographic
parameters must match.

Options:
  -h --help  Show this screen.
  --method=<classification_method>      Classification method to use: random-forest (for random
                                        forest) or svm (for support vector machines)
                                        [default: random-forest]
  --validation=<validation_data_path>   If given, it must be a path to a directory with vector data
                                        files (in shapefile format). These vectors must specify the
                                        target class of the validation pixels. A classification
                                        accuracy report is writen to stdout.
  --verbose                             If given, debug output is writen to stdout.

"""
import logging
import numpy as np
import os
import pickle
import sys

from docopt import docopt
from osgeo import gdal
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


logger = logging.getLogger(__name__)

# A list of "random" colors
COLORS = [
    "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
    "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
    "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
    "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
    "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
    "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
    "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
    "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
    "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
    "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
    "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
    "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
    "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
    "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
    "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
    "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"
]


def create_mask_from_vector(vector_data_path, cols, rows, geo_transform, projection, target_value=1,
                            output_fname='', dataset_format='MEM'):
    """
    Rasterize the given vector (wrapper for gdal.RasterizeLayer). Return a gdal.Dataset.

    :param vector_data_path: Path to a shapefile
    :param cols: Number of columns of the result
    :param rows: Number of rows of the result
    :param geo_transform: Returned value of gdal.Dataset.GetGeoTransform (coefficients for
                          transforming between pixel/line (P,L) raster space, and projection
                          coordinates (Xp,Yp) space.
    :param projection: Projection definition string (Returned by gdal.Dataset.GetProjectionRef)
    :param target_value: Pixel value for the pixels. Must be a valid gdal.GDT_UInt16 value.
    :param output_fname: If the dataset_format is GeoTIFF, this is the output file name
    :param dataset_format: The gdal.Dataset driver name. [default: MEM]

    """
    data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
    if data_source is None:
        report_and_exit("File read failed: %s", vector_data_path)
    layer = data_source.GetLayer(0)
    driver = gdal.GetDriverByName(dataset_format)
    target_ds = driver.Create(output_fname, cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])
    return target_ds


def vectors_to_raster(file_paths, rows, cols, geo_transform, projection):
    """
    Rasterize, in a single image, all the vectors in the given directory.

    The data of each file will be assigned the same pixel value. This value is defined by the order
    of the file in file_paths, starting with 1: so the points/poligons/etc in the same file will be
    marked as 1, those in the second file will be 2, and so on.

    :param file_paths: Path to a directory with shapefiles
    :param rows: Number of rows of the result
    :param cols: Number of columns of the result
    :param geo_transform: Returned value of gdal.Dataset.GetGeoTransform (coefficients for
                          transforming between pixel/line (P,L) raster space, and projection
                          coordinates (Xp,Yp) space.
    :param projection: Projection definition string (Returned by gdal.Dataset.GetProjectionRef)

    """
    labeled_pixels = np.zeros((rows, cols))
    for i, path in enumerate(file_paths):
        label = i + 1
        logger.debug("Processing file %s: label (pixel value) %i", path, label)
        ds = create_mask_from_vector(path, cols, rows, geo_transform, projection,
                                     target_value=label)
        band = ds.GetRasterBand(1)
        aux = band.ReadAsArray()
        logger.debug("Labeled pixels: %i", len(aux.nonzero()[0]))
        labeled_pixels += aux
        ds = None
    return labeled_pixels


def write_geotiff(fname, data, geo_transform, projection, data_type=gdal.GDT_Byte):
    """
    Create a GeoTIFF file with the given data.

    :param fname: Path to a directory with shapefiles
    :param data: Number of rows of the result
    :param geo_transform: Returned value of gdal.Dataset.GetGeoTransform (coefficients for
                          transforming between pixel/line (P,L) raster space, and projection
                          coordinates (Xp,Yp) space.
    :param projection: Projection definition string (Returned by gdal.Dataset.GetProjectionRef)

    """
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    dataset = driver.Create(fname, cols, rows, 1, data_type)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    ct = gdal.ColorTable()

    for pixel_value in range(len(classes) + 1):
        color_hex = COLORS[pixel_value]
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        ct.SetColorEntry(pixel_value, (r, g, b, 255))
    band.SetColorTable(ct)

    metadata = {
        'TIFFTAG_COPYRIGHT': 'CC BY 4.0',
        'TIFFTAG_DOCUMENTNAME': 'classification',
        'TIFFTAG_IMAGEDESCRIPTION': 'Supervised classification.',
        'TIFFTAG_MAXSAMPLEVALUE': str(len(classes)),
        'TIFFTAG_MINSAMPLEVALUE': '0',
        'TIFFTAG_SOFTWARE': 'Python, GDAL, scikit-learn'
    }
    dataset.SetMetadata(metadata)

    band.WriteArray(data)

    dataset = None  # Close the file
    return


def report_and_exit(txt, *args, **kwargs):
    logger.error(txt, *args, **kwargs)
    sys.exit(1)



def print_cm(cm, labels):
    """pretty print for confusion matrixes"""
    # https://gist.github.com/ClementC/acf8d5f21fd91c674808
    columnwidth = max([len(x) for x in labels])
    # Print header
    print(" " * columnwidth, end="\t")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end="\t")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("%{0}s".format(columnwidth) % label1, end="\t")
        for j in range(len(labels)):
            print("%{0}d".format(columnwidth) % cm[i, j], end="\t")
        print()


# ###############################################
# Starts testing code
# ###############################################
TRANSLATE_DICT = {
    'SJ': 1,
    'MZ': 2,
    'SRG': 3,
    'PN': 4,
    'MN': 5,
}


def test_method(vector_data_path, geo_transform, projection, target_value):
    """
    Rasterize our modified vector.
    """
    data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
    if data_source is None:
        report_and_exit("File read failed: %s", vector_data_path)
    layer = data_source.GetLayer(0)
    driver = gdal.GetDriverByName('MEM')
    target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds, [1], layer, options=["ATTRIBUTE=%s" % target_value])
    return target_ds


def predict_by_chunks(data, classifier):
    """
    Classify data by chunks.

    :param data:        An array of pixels.
    :param classifier:  A trained classifier.
    """

    # TODO: Improve this. Could be smarter. Maybe It could resolve how
    # to split data by itself.
    new_shape = (4, int(data.shape[0] / 4), 6)
    flat_pixels_divide = data.reshape(new_shape)

    result = np.array([])
    for i in range(flat_pixels_divide.shape[0]):
        logger.debug("Classifing [%i/%i]..." % (i, flat_pixels_divide.shape[0] - 1))
        result = np.concatenate([result, classifier.predict(flat_pixels_divide[i])])

    return result

def divide_test_and_training(all_data):
    rows, cols = all_data.shape
    sample = rows * cols
    # 50% of zeros
    amount_of_zeros = int((sample * 50) / 100)
    # 50% of ones
    amount_of_ones = sample - amount_of_zeros
    mask = np.array([0] * amount_of_zeros + [1] * amount_of_ones)
    # We want an array with 50% of 1's and 50% of 0's
    np.random.shuffle(mask)
    mask = mask.reshape(all_data.shape)

    training = all_data * mask

    # if mask = [0, 1, 0] ==> ((mask - 1) * (-1)) = [1, 0, 1]
    test = all_data * ((mask - 1) * (-1))

    return test, training


def delete_extra_fields_from_vector(vector_data_path):
    # Open for update
    ds = gdal.OpenEx(vector_data_path, gdal.OF_UPDATE )
    if ds is None:
        print("Open failed.")
        sys.exit( 1 )

    lyr = ds.GetLayer()

    # Comienza a leer desde el primer feature
    lyr.ResetReading()

    lyr_defn = lyr.GetLayerDefn()
    # Get field by name
    field_roi_e_14_1 = lyr_defn.GetFieldIndex('ROI_E_14_1')
    field_area = lyr_defn.GetFieldIndex('Area')
    field_e_2015 = lyr_defn.GetFieldIndex('E_2015')

    lyr.DeleteField(field_area)
    lyr.DeleteField(field_e_2015)
    # Close vector
    ds = None


def translate_strings_to_int(vector_data_path):
    ds = gdal.OpenEx(vector_data_path, gdal.OF_UPDATE)
    if ds is None:
        print("Open failed.")
        sys.exit(1)
    lyr = ds.GetLayer()

    lyr.ResetReading()
    lyr_defn = lyr.GetLayerDefn()
    field_roi_e_14_1 = lyr_defn.GetFieldIndex('ROI_E_14_1')
    for feat in lyr:
        field_key = feat.GetField(field_roi_e_14_1)
        feat.SetField(field_roi_e_14_1, TRANSLATE_DICT[field_key])
        lyr.SetFeature(feat)
    ds = None
# ###############################################
# End testing code
# ###############################################


if __name__ == "__main__":
    opts = docopt(__doc__)

    raster_data_path = opts["<input_fname>"]
    train_data_path = opts["<train_data_path>"]
    output_fname = opts["<output_fname>"]
    validation_data_path = opts['--validation'] if opts['--validation'] else None
    log_level = logging.DEBUG if opts["--verbose"] else logging.INFO
    method = opts["--method"]

    logging.basicConfig(level=log_level, format='%(asctime)-15s\t %(message)s')
    gdal.UseExceptions()

    logger.debug("Reading the input: %s", raster_data_path)
    try:
        raster_dataset = gdal.Open(raster_data_path, gdal.GA_ReadOnly)
    except RuntimeError as e:
        report_and_exit(str(e))

    geo_transform = raster_dataset.GetGeoTransform()
    proj = raster_dataset.GetProjectionRef()
    bands_data = []
    for b in range(1, raster_dataset.RasterCount + 1):
        band = raster_dataset.GetRasterBand(b)
        bands_data.append(band.ReadAsArray())

    bands_data = np.dstack(bands_data)
    rows, cols, n_bands = bands_data.shape
    # A sample is a vector with all the bands data. Each pixel (independent of its position) is a
    # sample.
    n_samples = rows * cols

    logger.debug("Process the training data")
    try:
        files = [f for f in os.listdir(train_data_path) if f.endswith('.shp')]
        classes = [f.split('.')[0] for f in files]
        shapefiles = [os.path.join(train_data_path, f) for f in files if f.endswith('.shp')]
    except OSError.FileNotFoundError as e:
        report_and_exit(str(e))

    labeled_pixels = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)

    test_data, training_data = divide_test_and_training(labeled_pixels)

    # is_train = np.nonzero(labeled_pixels)
    is_train = np.nonzero(training_data)
    training_labels = labeled_pixels[is_train]
    training_samples = bands_data[is_train]

    flat_pixels = bands_data.reshape((n_samples, n_bands))

    # min_label_value = int(labeled_pixels.min())
    # max_label_value = int(labeled_pixels.max())
    # pixels_per_class = {}

    # # Collect some useful data
    # for label in range(min_label_value, max_label_value + 1):
    #     key = label
    #     if label >= 1 and label <= len(classes):
    #         # We want the "filename" as key.
    #         key = classes[label - 1]
    #     pixels_per_class[key] = labeled_pixels[labeled_pixels == label].shape[0]

    #
    # Perform classification
    #
    CLASSIFIERS = {
        # http://scikit-learn.org/dev/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        'random-forest': RandomForestClassifier(n_jobs=4, n_estimators=10, class_weight='balanced'),
        # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        'svm': SVC(class_weight='balanced')
    }

    classifier = CLASSIFIERS[method]
    logger.debug("Train the classifier: %s", str(classifier))
    classifier.fit(training_samples, training_labels)

    # logger.debug("Saving trained object and some extra information...")
    # pixel_data = {}
    # pixel_data['pixel_to_classify'] = flat_pixels
    # pixel_data['cols'] = cols
    # pixel_data['rows'] = rows
    # with open('classifier_trained.pickle', 'wb') as fclass:
    #     with open('pixels_data.pickle', 'wb') as fpixels:
    #         pickle.dump(classifier, fclass)
    #         pickle.dump(pixel_data, fpixels)

    result = predict_by_chunks(flat_pixels, classifier)

    # Reshape the result: split the labeled pixels into rows to create an image
    classification = result.reshape((rows, cols))
    write_geotiff(output_fname, classification, geo_transform, proj)
    logger.info("Classification created: %s", output_fname)

    for_verification = np.nonzero(test_data)
    verification_labels = labeled_pixels[for_verification]
    predicted_labels = classification[for_verification]

    logger.info("Confussion matrix:\n")
    print_cm(metrics.confusion_matrix(verification_labels, predicted_labels), classes)
    target_names = ['Class %s' % s for s in classes]
    logger.info("Classification report:\n%s",
                metrics.classification_report(verification_labels, predicted_labels,
                                              target_names=target_names))
    logger.info("Classification accuracy: %f",
                metrics.accuracy_score(verification_labels, predicted_labels))

    #
    # Validate the results
    #
    if validation_data_path:
        logger.debug("Process the verification (testing) data")
        try:
            shapefiles = [os.path.join(validation_data_path, "%s.shp" % c) for c in classes]
        except OSError.FileNotFoundError as e:
            report_and_exit(str(e))

        verification_pixels = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)
        for_verification = np.nonzero(verification_pixels)
        verification_labels = verification_pixels[for_verification]
        predicted_labels = classification[for_verification]

        logger.info("Confussion matrix:\n%s", str(
            metrics.confusion_matrix(verification_labels, predicted_labels)))
        target_names = ['Class %s' % s for s in classes]
        logger.info("Classification report:\n%s",
                    metrics.classification_report(verification_labels, predicted_labels,
                                                  target_names=target_names))
        logger.info("Classification accuracy: %f",
                    metrics.accuracy_score(verification_labels, predicted_labels))
