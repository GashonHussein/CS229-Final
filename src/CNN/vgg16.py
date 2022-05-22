# import keras
import numpy as np, tensorflow as tf
import pathlib
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
# from storage.get_images import *

###Follow the tutorial:https://www.tensorflow.org/tutorials/load_data/images###


def data_preprocess(dataset_url):
    data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                    fname='drowsiness',
                                    untar=True)
    data_dir = pathlib.Path(data_dir)
    print(data_dir)

    image_count = len(list(data_dir.glob('*/*.png')))
    print("Total number of images:", image_count)

    batch_size = 32
    img_height = 180
    img_width = 180

    train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    return train_ds, val_ds
    
def model_build(VGG16):
    if VGG16:   
        vggmodel = VGG16(weights='imagenet', include_top=True)
        X = vggmodel.layers[-2].output
        predictions = tf.keras.layers.Dense(2,activation='softmax')(X)
        model = tf.keras.models.Model(input=vggmodel.input, output=predictions)
    else:
        num_classes = 2
        model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255), # normalization layer to standardize values to be in the [0, 1] range 
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
        ])  

    return model

def main(VGG16):
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    # dataset_url = "https://storage.googleapis.com/cs-229-storage-bucket/drowsiness/raw"

    train_ds, val_ds = data_preprocess(dataset_url)
    
    model = model_build(VGG16)
    
    model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

    model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3)

if __name__ == "__main__":
    VGG16 = True # use VGG16 or self-defined CNN
    main(True)
