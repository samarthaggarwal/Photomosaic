# Photomosaic

This repository contains the code to construct the photomosaic of an image using a flower dataset.

### Dataset
[Description of Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/) \
[Flower102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) : Flower dataset consisting of 102 classes. \
[Flower17](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html) : You can alternatively use the flower dataset with 17 classes.

### Parameters
`infile` : path to target image \
`outfile` : path to output image \
`tol` : tolerance, if tol=5 then every patch in target image will be replaced randomly by one of the top 5 matches from the library \
`f` : scale, every library image replaces an fxf patch of the target image. Lower f corresponds to higher resolution of the output image.

### Usage
1. `./setup.sh`
2. `python mosaic.py --infile images/input/uiuc.jpg --outfile images/output/uiuc.png --tol 5 --f 16` \
Replace `infile`, `outfile` with your image paths. Tweak `tol` and `f` until you get the desired output.


