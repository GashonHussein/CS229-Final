# import keras
import numpy as np, tensorflow as tf
import pathlib
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from storage.get_images import *

vggmodel = VGG16(weights='imagenet', include_top=True)



# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
dataset_url = "https://storage.googleapis.com/cs-229-storage-bucket/drowsiness/raw.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='drowsiness',
                                   untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.png')))
print(image_count)

# for i in range(0, 11, 10):
#     res = get_images_url(i)
#     n_y = len(res)
#     y = np.full(n_y,i)
    # print(type(res))
