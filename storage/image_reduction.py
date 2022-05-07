import numpy as np
import cv2
import json
from urllib.request import urlopen
from get_images import *
import matplotlib.pyplot as plt



data = {"0": {}, "5": {}, "10": {}}

for i in range(0, 11, 5):
    res = get_images_url(i)
    for path in res:
        req = urlopen(res[0])
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1) # 'Load it as it is'
        # img = io.imread(path)
        resized = cv2.resize(img, (90,120))

# plt.imshow(resized)
# plt.show()
        