# import keras
import numpy as np, tensorflow as tf
import pathlib
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
# from keras.preprocessing.image import ImageDataGenerator
# from tensorflow.python.lib.io import file_io
# import tensorflow_cloud as tfc
# import tensorflow_datasets as tfds

# from storage.get_images import *

###Follow the tutorial:https://www.tensorflow.org/tutorials/load_data/images###


def data_preprocess(dataset_url):

    # (ds_train, ds_test), metadata = tfds.load(
    # dataset,
    # split=["train", "test"],
    # shuffle_files=True,
    # with_info=True,
    # as_supervised=True,
    # )
 
    # NUM_CLASSES = metadata.features["label"].num_classes

    # print(ds_train)
    # image = dataset_url.read()
    # print(image)

    data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                    fname='images',
                                    untar=True)
    data_dir = pathlib.Path(data_dir)
    # print(data_dir)

    # all_files = data_dir.glob("*/*.png")
    # dot_files= data_dir.glob("*/.*.png")
    # complete_dir = set(all_files)-set(dot_files)

    image_count = len(list(data_dir.glob('*/*.png')))
    print("Total number of images:", image_count)

    # batch_size = 32
    img_height = 120
    img_width = 90

    complete_dir = data_dir

    train_ds = tf.keras.utils.image_dataset_from_directory(
    complete_dir,
    validation_split=0.2,
    shuffle=True,
    subset="training",
    seed=133,
    image_size=(img_height, img_width),
    batch_size=16)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    complete_dir,
    validation_split=0.2,
    shuffle=True,
    subset="validation",
    seed=133,
    image_size=(img_height, img_width),
    batch_size=16)

    # print(train_ds)
    class_names = train_ds.class_names
    print("class list",class_names)

    # for image_batch, labels_batch in train_ds:
    #     print(image_batch.shape)
    #     print(labels_batch.shape)
    #     break

    # AUTOTUNE = tf.data.AUTOTUNE

    # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds
    
def model_build(VGG):
    if VGG:   
        vggmodel = VGG16(weights='imagenet', include_top=True)
        X = vggmodel.layers[-2].output
        predictions = tf.keras.layers.Dense(2,activation='softmax')(X)
        model = tf.keras.models.Model(vggmodel.input, predictions)
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
        # tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes)
        ])  

    return model

def main(VGG16):
    # dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    dataset_url = "https://storage.googleapis.com/cs-229-storage-bucket/production/raw_images.tgz"
   
    train_ds, val_ds = data_preprocess(dataset_url)
   
    model = model_build(VGG16)
    # input = tf.keras.Input(shape=([120,90,3]))
    # output = model(input)

    model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

    model.summary()

    history = model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch = 10,
    epochs=30)



    np.save('CNN_history.npy', history.history)
    
    # test_loss, test_acc = model.evaluate(val_image, val_labels, verbose=2)

    # print('\nTest accuracy:', test_acc)

    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']

    
if __name__ == "__main__":
    VGG = True # use VGG16 or self-defined CNN
    main(VGG)

