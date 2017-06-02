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
import requests

local_filename, headers = urllib.request.urlretrieve("http://farm1.static.flickr.com/101/289376763_4f2dee00c9.jpg")
f = io.imread(local_filename)
f.shape
f.dtype

local_filename, headers = urllib.request.urlretrieve("https://www.dogbreedinfo.com//images3/Goldenellie.jpg")
f = io.imread(local_filename)
f.shape
f.dtype

r = requests.get("https://www.dogbreedinfo.com//images3/Goldenellie.jpg")
r.content
from io import BytesIO
f = io.imread(BytesIO(r.content))

#Key 1: 3205df368d50445ca01087388f1c8b8b
#Key 2: 27d16a4a969d48bda120c0e7420b09a4
headers = {
    'Content-Type': 'multipart/form-data',
    'Ocp-Apim-Subscription-Key': '3205df368d50445ca01087388f1c8b8b',
}
param = urllib.parse.urlencode({
    'q': 'golden retriever',
})
#https://docs.microsoft.com/en-us/azure/cognitive-services/bing-image-search/search-the-web
#https://dev.cognitive.microsoft.com/docs/services/56b43f0ccf5ff8098cef3808/operations/571fab09dbe2d933e891028f
r = requests.post("https://api.cognitive.microsoft.com/bing/v5.0/images/search?%s" % param, headers=headers)
r.status_code
r.raise_for_status()
r.content
r.json()
r.text
r.encoding

r_json = r.json()
r_json.keys()
r_json.values()

len(r_json)
r_json['webSearchUrl']
r_json['totalEstimatedMatches']
type(r_json['value'])
len(r_json['value'])
r_json['value'][0]['contentUrl']




params = urllib.parse.urlencode({
    'q': 'golden retriever',
    'count': '100',
    'offset': '0',
    'mkt': 'en-us',
    #'safeSearch': 'Moderate',
    'safeSearch': 'Off',
})
r = requests.get("https://api.cognitive.microsoft.com/bing/v5.0/images/search?%s" % params, headers=headers)
r.json()['webSearchUrl']

len(r.json()['value'])
r.json()['value'][0]
r.json()['value'][99]['contentUrl']
r.json()['value'][99]['hostPageUrl']

link = r.json()['value'][0]['contentUrl']
local_filename, headers = urllib.request.urlretrieve(link)
f = io.imread(local_filename)
f.shape
f.dtype


r = requests.get(r.json()['value'][0]['contentUrl'])
r.history
r.is_redirect
r.ok
r.headers
r.request.headers
for resp in r.history:
    print(resp.url)

link = r.json()['value'][99]['contentUrl']
o = urllib.parse.urlparse(link)
o.geturl()
urllib.parse.urlsplit(link)
urllib.parse.urldefrag(link)


import re

len(link)
match_object = re.search('r=(http.+)&', link)
redirect = match_object.group(1)
redirect
redirect = re.sub('%2f', '/', redirect)
redirect = re.sub('%3a', ':', redirect)
redirect

re.sub(r'([\w\.-]+)@([\w\.-]+)', r'\1@yo-yo-dyne.com', str)