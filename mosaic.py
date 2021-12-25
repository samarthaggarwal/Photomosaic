import numpy as np
from matplotlib import pyplot as plt
from skimage import io
import cv2
from random import randint
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import time
import pickle
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--infile", default="images/input/uiuc.jpg")
parser.add_argument("--outfile", default="images/output/uiuc.png")
parser.add_argument("--libpath", default="flowers102/")
parser.add_argument("--tol", default=3)
parser.add_argument("--f", default=32)
args = parser.parse_args()

print("infile: {}\noutfile: {}\ntol: {}\nf: {}".format(args.infile, args.outfile, args.tol, args.f ))

def plot_images(titles, images, size=4, axis=False):
  n = len(titles)
  fig, axs = plt.subplots(1,n, figsize=(n*size,size))
  if n < 2:
    axs = [axs]
  if not axis:
    _ = [ax.axis("off") for ax in axs]
  _ = [axs[i].set_title(titles[i]) for i in range(n)]
  _ = [axs[i].imshow(images[i]) if images[i].ndim == 3 else axs[i].imshow(images[i], cmap="gray") for i in range(n)]


def mean_color(image):
  return np.mean(image, axis=(0,1))

def library_mean_color(images):
    return np.mean(images, axis=(1,2))

def plot_image(image):
    plt.imshow(image.reshape(1,1,3)/255.0)


# # Simplest Photomosaic 
# 1. Downsample target image by a factor of F where the mosaic will be made of up of images of size FxF. 
# 2. Each pixel of the downsampled target image will represent FxF patch in original image
# 3. Search library of images to find image that has average color (mean color) closest to the pixel in the downsampled image
# 4. Resize the library image to FxF and put it into the output image numpy array


def simple_mosaic(target, images, images_color, f=32, tol=1):
    h,w = target.shape[:2]
    resized_h, resized_w = h//f, w//f
    resized_target = cv2.resize(target, (resized_w, resized_h))
    
    mosaic = np.zeros( ((target.shape[0]//f)*32, (target.shape[1]//f)*32, 3))
    for r in tqdm(range(resized_h)):
        for c in range(resized_w):
            target_color = resized_target[r,c]
            distances = np.sum(np.square(images_color - target_color), axis=1)
#             closest = np.argmin(distances)
            nearest = np.argsort(distances)
            idx = np.random.randint(tol)
            closest = nearest[idx]
            mosaic[r*32:(r+1)*32, c*32:(c+1)*32,:] = images[closest]
    return mosaic/255.


##### Preprocess Data #####

start_time = time.time()

pkl_file = Path(args.libpath + "/" + "library.pkl")
if pkl_file.exists():
	# load library from pickle file
	with open(pkl_file, 'rb') as f:
		pkl = pickle.load(f)
	library = pkl['library']
	library_color = pkl['library_color']

else:
	onlyfiles = [f for f in listdir(args.libpath) if isfile(join(args.libpath, f)) and "jpg" in f]
	onlyfiles.sort()
	#onlyfiles = onlyfiles[:1000]
	
	library = []
	for f in tqdm(onlyfiles):
	    img = io.imread(join(args.libpath, f))
	    img = cv2.resize(img, (32,32))
	    library.append(img)
	library = np.stack(library, axis=0)
	#print(library.shape)
	
	plot_images([str(i) for i in range(10)], library[:10])
	plt.imshow(library[0])
	
	library_color = library_mean_color(library)
	#print(library_color.shape)

	# save library to pickle file
	pkl = {'library':library, 'library_color':library_color}
	with open(pkl_file, 'wb') as f:
		pickle.dump(pkl, f)

# Plot sample images  from library 
sample_images = [library[i] for i in range(10)]
sample_labels = [str(i) for i in range(10)]
#plot_images(sample_labels, sample_images, size=1)
sample_mean_colors = library_color[:10].reshape(-1,1,1,3)/255.
#plot_images(sample_labels, sample_mean_colors, size=1)

end_time = time.time()
print("preprocessing time = {}sec".format(round(end_time - start_time, 2)))

##### Composing Target Mosaic #####

start_time = time.time()

target = io.imread(args.infile)
mosaic = simple_mosaic(target, library, library_color, f=int(args.f), tol=int(args.tol))
#print(mosaic.shape)
#plot_images(['original','mosaic'], [target,mosaic],size=8)

plt.imsave(args.outfile, mosaic)

end_time = time.time()
print("mosaic time = {}sec".format(round(end_time - start_time, 2)))



