import sys
import skimage.io as io
import numpy as np
import json
import jsonpickle
from json import JSONEncoder

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

sys.path.append('../storage')
from get_images import get_images_url

# Use the application default credentials
cred = credentials.Certificate("./util/service-account.json")
firebase_admin.initialize_app(cred)


db = firestore.client()

# jsonpickle.decode(data[query_name]) to decode
def upload_array(query_name, arr, classification, data):
    arr_str = np.array2string(arr)
    data["{}".format(classification)][query_name] = arr_str

    # with open('./util/data.json', 'w') as outfile:
    #     json.dump(data, outfile)
        
    print(arr)
    print("uploading",query_name)
    db.collection('{}'.format(classification)).add({
        u'feature_1': arr_str,
        u'url': query_name,
    })



# this function is called for each image in the database
def process_image(image_url, classification, data):
    # Read
    stackColors = io.imread('single_color_test_image.png')
    # Split
    red_features_matrix = stackColors[:, :, 0]
    green_features_matrix = stackColors[:, :, 1]
    blue_features_matrix = stackColors[:, :, 2]
    # For CNN Keep channels in place
    cnn_features_stack = stackColors[:, :, :3]
    # For other models (logistic regression) flatten features/channels
    flattened_features = cnn_features_stack.reshape(1, -1)
    upload_array(image_url, flattened_features[0], classification, data)
    return flattened_features[0]
    
    # all_samples_stackColors = np.array([])
    # for image_url in image_urls:
    #     stackColors = io.imread('single_color_test_image.png')
    #     RGB_channels_only = stackColors[:, :, :3]
    #     all_samples_stackColors = np.append(all_samples_stackColors, RGB_channels_only)
    # # Split
    # # Shapes: (num_images, image_dim_h, image_dim_w)
    # all_red_features_matrix = all_samples_stackColors[:, :, :, 0]
    # all_green_features_matrix = all_samples_stackColors[:, :, :, 1]
    # all_blue_features_matrix = all_samples_stackColors[:, :, :, 2]
    # # For CNN Keep channels in place
    # # Shape: (num_images, image_dim_h, image_dim_w, 3) -> Same as all_samples_stackColors
    # all_cnn_features_stack = all_samples_stackColors
    # # For other models (logistic regression) flatten features/channels
    # # Shape: (num_images, image_dim_h * image_dim_w * 3)
    # all_flattened_features = cnn_features_stack.reshape(np.shape(all_cnn_features_stack)[0], -1)
    # print(all_flattened_features)

#testing one image
def test_one(data):
    url = "https://storage.googleapis.com/cs-229-storage-bucket/drowsiness/raw/10/qhqk03.png"
    stackColors = io.imread(url)
    # Split
    red_features_matrix = stackColors[:, :, 0]
    green_features_matrix = stackColors[:, :, 1]
    blue_features_matrix = stackColors[:, :, 2]
    # For CNN Keep channels in place
    cnn_features_stack = stackColors[:, :, :3]
    # For other models (logistic regression) flatten features/channels
    flattened_features = cnn_features_stack.reshape(1, -1)
    upload_array(url, flattened_features[0], 0, data)
    return flattened_features

def main():
    data = {"0": {}, "5": {}, "10": {}}
    print(test_one(data))
    # all_flattened_features_by_class = np.array([])
    # for i in range(0, 11, 5):
    #     res = get_images_url(i)
    #     curr_flattened_features = np.array([])
    #     for image in res:
    #         image_features = process_image(image, i, data)
    #         curr_flattened_features = np.append(curr_flattened_features, image_features)
    #     all_flattened_features_by_class = np.append(all_flattened_features_by_class, curr_flattened_features)
    with open('./util/data.json', 'w') as outfile:
        json.dump(data, outfile)


if __name__ == "__main__":
    main()
