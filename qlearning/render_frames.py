from __future__ import unicode_literals
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
plt.ion()

files = sorted(os.listdir('../models/gif')[1:], key=lambda x: int(x.split('-')[1][0:-4]))
frames = []

for file in files:
    arr = np.load('../models/gif/{}'.format(file))
    plt.imshow(arr)
    plt.pause(0.1)
    frames.append(Image.fromarray(arr))

with open('../another_run.gif', 'wb') as f:  # change the path if necessary
    im = Image.new('RGB', frames[0].size)
    im.save(f, save_all=True, append_images=frames)