import os
import gc
import psutil
import skimage
import numpy as np
import pandas as pd
from skimage import data, io

img = skimage.data.hubble_deep_field()
img.size
img.shape

psutil.cpu_percent()
psutil.cpu_stats()
psutil.cpu_count()

if os.path.exists("H:\python-notes"):
    os.chdir("H:\python-notes")
print(os.getcwd())

gc.collect()
starting_mem = psutil.virtual_memory().used

IMAGES = []
for row in np.arange(1000):
    mem = psutil.virtual_memory()
    mem_used = (mem.used - starting_mem) / (1024 * 1024 * 1024)
    if row % 10 == 0:
        print(row, round(mem_used, 3) , 'GB')
    IMAGES.append(('test', 123456, skimage.img_as_float(img)))
    #IMAGES.append(skimage.img_as_float(img))
    #IMAGES.append(98.765435435)
    if mem_used > 10:
        df = pd.DataFrame(IMAGES, columns=['dir', 'ID', 'image'])
        df.to_csv('test_file.csv', header=False, mode='a')
        IMAGES.clear()
        del df
        gc.collect()
        print('memory freed')
df = pd.DataFrame(IMAGES, columns=['dir', 'ID', 'image'])
df.to_csv('test_file.csv', header=False, mode='a')

gc.collect()
mem = psutil.virtual_memory()
mem.available
mem.percent

#GB
mem.used /  (1024 * 1024 * 1024)
del IMAGES

gc.collect()
mem = psutil.virtual_memory()
mem.used /  (1024 * 1024 * 1024)