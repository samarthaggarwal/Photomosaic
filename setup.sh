#!/bin/bash

wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
wget --output-document=images/input/uiuc.jpg 'https://cdn.vox-cdn.com/thumbor/F8MeAfakEGVRVeJfMs4_0mO4YvU=/0x0:920x613/1200x800/filters:focal(387x234:533x380)/cdn.vox-cdn.com/uploads/chorus_image/image/67344119/university_of_illinois_campus.0.0.jpg' 
echo "downloaded"

tar -xzf 102flowers.tgz
mv jpg flowers102
echo "unzipped"

exit 0

