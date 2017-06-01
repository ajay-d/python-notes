import os
import skimage
import numpy as np
from skimage import data, io
filename = os.path.join(skimage.data_dir, 'moon.png')

moon = io.imread(filename)

#http://scikit-image.org/docs/dev/user_guide/numpy_images.html
type(moon)
moon.shape
moon.size
moon.min(), moon.max()
np.unique(moon)
moon.dtype

#http://scikit-image.org/docs/dev/user_guide/data_types.html#data-types
from skimage import img_as_float, img_as_int
print(img_as_float(moon))
img_as_float(moon).min(), img_as_float(moon).max()
img_as_int(moon).min(), img_as_int(moon).max()

import urllib

local_filename, headers = urllib.request.urlretrieve("http://farm1.static.flickr.com/101/289376763_4f2dee00c9.jpg")
f = io.imread(local_filename)
f.shape
f.dtype

local_filename, headers = urllib.request.urlretrieve("https://www.dogbreedinfo.com//images3/Goldenellie.jpg")
f = io.imread(local_filename)
f.shape
f.dtype

