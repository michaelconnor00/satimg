# Setup Requirements for running Demo



The easiest way to run the software used in my project is to Install Anaconda

* Go to https://www.continuum.io/downloads and find correct package. Recommend using Python 3.5, likely 64bit.
    - Run the bash command shown on this page.
    - Let it install to your home directory.
    - Select yes for prepending anaconda to your path.


Now install all the required dependencies from there. The following are steps for setting up the environment:


```bash
# Create the env
conda create --name osgeoenv python=3.5
# Activate the env
source activate osgeoenv

# Install dependencies
conda install -c conda-forge gdal
conda install -c conda-forge -c rios arcsi
conda install -c conda-forge scikit-learn
conda install -c conda-forge scikit-image
conda install -c conda-forge matplotlib
conda install -c conda-forge jupyter
```



Do a quick test to check if the env/bin has been added to the PATH env var.

```bash
# Linux/Unix
$ which gdalinfo
/anaconda/envs/osgeoenv/bin/gdalinfo  # This path may vary depending on Operating System
```
