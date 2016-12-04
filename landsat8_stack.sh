#! /bin/bash
set -e
  #statements
FILE=$1

BASE=$(basename $FILE)

LEN_FILE=${#FILE}
LEN_BASE=${#BASE}
DIR_LEN=$((LEN_FILE-LEN_BASE))

cd ${FILE:0:DIR_LEN}

FILE_NO_EXT=$((${#BASE}-7))

#establish an id variable
id=${BASE:0:FILE_NO_EXT}

#use a for loop to reproject each of the bands you will be working with.
for BAND in {4,3,2}
do
  gdalwarp -t_srs EPSG:4326 $id"_B"$BAND.tif $BAND-projected.tif
  #translate each of your bands into the 8-bit format with default settings of -ot and -scale
  gdal_translate -ot Byte -scale 0 65535 0 255 $BAND-projected{,-scaled}.tif
done

#merge the three reprojected band images into a single composite image
gdal_merge.py -v -ot Byte -separate -of GTiff -co PHOTOMETRIC=RGB -o $id-RGB-scaled.tif 2-projected-scaled.tif 3-projected-scaled.tif 4-projected-scaled.tif 5-projected-scaled.tif 6-projected-scaled.tif 7-projected-scaled.tif

#color corrections in blue bands to deal with haze factor,
#and across all bands for brightness, contrast and saturation
convert -channel B -gamma 1.05 -channel All -sigmoidal-contrast 20,20% -modulate 100,150 $id-RGB-scaled.tif $id-RGB-scaled-cc.tif

#use a cubic downsampling method to add overview
#(other interpolation methods are available)
gdaladdo -r cubic $id-RGB-scaled-cc.tif 2 4 8 10 12

#call the TIFF worldfile for the requested image,
#change name of file to match file needing georeference,
#and apply georeference
listgeo -tfw 3-projected.tif
mv 3-projected.tfw $id-RGB-scaled-cc.tfw
gdal_edit.py -a_srs EPSG:3857 $id-RGB-scaled-cc.tif

#remove black background
gdalwarp -srcnodata 0 -dstalpha $id-RGB-scaled-cc.tif $id-RGB-scaled-cc-2.tif

for BAND in {2,3,4}
do
  rm $BAND-projected.tif
  rm $BAND-projected-scaled.tif
done
