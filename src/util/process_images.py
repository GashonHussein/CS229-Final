import sys
import skimage.io as io
import numpy as np
import json
import jsonpickle
from json import JSONEncoder
import matplotlib.pyplot as plt 

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

sys.path.append('../storage')
from get_images import get_images_url
from augment_data import augment_data

# Use the application default credentials
cred = credentials.Certificate("./util/service-account.json")
firebase_admin.initialize_app(cred)


db = firestore.client()

# jsonpickle.decode(data[query_name]) to decode
def upload_array(query_name, arr, classification, data):
    arr_str = arr.tolist()
    data["{}".format(classification)][query_name] = arr_str

    # with open('./util/data.json', 'w') as outfile:
    #     json.dump(data, outfile)
    print("uploading", query_name)
    # db.collection('{}'.format(classification)).add({
    #     u'feature_1': arr_str,
    #     u'url': query_name,
    # })

# this function is called for each image in the database
def process_image(image_url, classification, data):
    # Read
    stackColors = io.imread(image_url)
    # Split
    red_features_matrix = stackColors[:, :, 0]
    green_features_matrix = stackColors[:, :, 1]
    blue_features_matrix = stackColors[:, :, 2]
    # For CNN Keep channels in place
    cnn_features_stack = stackColors[:, :, :3]
    # For other models (logistic regression) flatten features/channels
    flattened_features = cnn_features_stack.reshape(1, -1)
    upload_array(image_url, flattened_features[0], classification, data)
    return flattened_features

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

def total_image_count(arr):
    total = 0
    for i in arr:
        total += len(get_images_url(i))
    return total
    

def main():
    data = {"0": {}, "5": {}, "10": {}}
    #print(test_one(data))
    # all_flattened_features_by_class = np.array([])
    # for i in range(0, 11, 5):
    #     res = get_images_url(i)
    #     curr_flattened_features = np.array([])
    #     for image in res:
    #         image_features = process_image(image, i, data)
    #         curr_flattened_features = np.append(curr_flattened_features, image_features)
    #     all_flattened_features_by_class = np.append(all_flattened_features_by_class, curr_flattened_features)

    all_flattened_features_by_class = []
    all_flattened_class = None

    all_classification = [0, 10]
    image_count = total_image_count(all_classification)
    curr_count = 0
    for i in all_classification:
        res = get_images_url(i)
        curr_flattened_features = []
        n = len(res)
        for j in range(n): # grab n images per classificaiton
            print("processing classification {} image {} completion: {}%".format(i, j, round(curr_count * 100 / image_count, 3)))
            image_features = process_image(res[j], i, data)[0]
            # print("image features", image_features)
            curr_flattened_features.append(np.array(image_features)) # may need to change to regular
            curr_count += 1
            # print(type(curr_flattened_features))
        
        all_flattened_features_by_class.append(curr_flattened_features) 
   
        # all_flattened_features_by_class = np.append(all_flattened_features_by_class, curr_flattened_features) 
    # print("writing data to ./util/output.json")
    # with open('./util/output.json', 'w') as outfile:
    #     json.dump(data, outfile)

    logistic_acc_metrics = logistic_regression_full(all_flattened_features_by_class)
    print(logistic_acc_metrics)
    #plt.show()
    


if __name__ == "__main__":
    main()

def logistic_regression(all_train_data_input, all_train_data_classification, batch_size, epochs, learning_rate = 0.05, W = None, b = None):
    
    num_examples = np.shape(all_train_data_input)[1]
    num_features = np.shape(all_train_data_input)[0]
    if batch_size == num_examples:
        train_batches = [all_train_data_input]
        train_classes = [all_train_data_classification]
    else:
        train_batches = np.split(all_train_data_input, batch_size)
        train_classes = np.split(all_train_data_classification, batch_size)
    
    if W is None:
        W = np.zeros((1, num_features))
    if b is None:
        b = np.zeros((1,1))
    
    def sigmoid(z):
        return 1/(1 + np.exp(-z))
    
    def calc_Z(W, b, x):
        return b + np.dot(W, x)
    
    def calc_dZ(A, Y):
        return A-Y
    
    def update_W(dZ, x, W):
        dW = np.dot(dZ, x.T)
        return W - learning_rate * dW
    
    def update_b(dZ, b):
        db = np.sum(dZ, axis = 1)
        return b - learning_rate * db
    
    loss_arr = np.array([])
    
    for curr_epoch in range(epochs):
        for index, train_batch in enumerate(train_batches):
            Y = train_classes[index]
            Z = calc_Z(W, b, train_batch)
            A = sigmoid(Z)
            
            dZ = calc_dZ(A, Y)
            
            W = update_W(dZ, train_batch, W)
            b = update_b(dZ, b)
        loss_arr = np.append(loss_arr, -np.sum(Y * np.log(A) + (1-Y) * np.log(1 - A)))
    plt.plot(np.arange(epochs), loss_arr)  
    return W, b


def logistic_regression_prediction(all_test_data_input, all_test_data_classification, W = None, b = None):
    
    num_test_examples = np.shape(all_test_data_input)[1]
    num_input_features = np.shape(all_test_data_input)[0]
    
    X = all_test_data_input
    
    def sigmoid(input_arr):
        return 1/(1 + np.exp(-input_arr))
    
    def calc_Z(W, b, x):
        return b + np.dot(W, x)
    
    Z = np.dot(W, X) + b
    A = sigmoid(Z)
    
    Y_hat = np.copy(A)
    Y_hat[A >= 0.5] = 1
    Y_hat[A < 0.5] = 0
    
    return Y_hat, A

# Get and print Negative Log Liekhlood loss and return it
def NLL_loss_calc(A, Y):
    num_test_examples = np.shape(A)[1]
    total_NLL_loss = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    average_NLL_loss = total_NLL_loss/num_test_examples
    
    print("Total Loss over " + str(num_test_examples) + " test examples: " + str(total_NLL_loss))
    return total_NLL_loss

# Calculate and print accuracy and return dictionary of accuracy metrics
def accuracy_calc(Y_hat, Y):
    num_test_examples = np.shape(Y_hat)[1]
    
    Total_P = np.sum(Y_hat)
    Total_N = num_test_examples - Total_P
    
    diff_vec = Y_hat - Y
    
    False_P = np.sum(diff_vec[Y_hat == 1])
    True_P = Total_P - False_P
    False_N = np.sum(abs(diff_vec[Y_hat == 0]))
    True_N = Total_N - False_N
    
    incorrect = np.sum(abs(Y - Y_hat))
    correct = num_test_examples - incorrect
    
    precision = True_P/(Total_P)
    recall = True_P/(True_P + False_N)
    F1_Score = 2 * precision * recall/(precision + recall)
    accuracy = correct/num_test_examples
    
    # assert incorrect == False_P + False_N
    # assert correct == True_P + True_N
    
    print("Overall accuracy over " + str(num_test_examples) + " test examples: " + str(accuracy))
    
    return {"F1_Score" : F1_Score, "precision" : precision, "recall" : recall, "accuracy" : accuracy,
           "Total_Positive" : Total_P, "Total_Negative" : Total_N, "total_correct" : correct,
           "total_incorrect" : incorrect, "False_Positves" : False_P, "True_Positives" : True_P,
           "False_Negatives" : False_N, "True_Negatives" : True_N}
        
# Take all the data and return shuffled x and y data pairs
def x_y_data_create(all_data):
    num_examples = 0
    
    # Calculate total examples in data
    # print("here", all_data)
    for curr_data in all_data:
        # print(curr_data)
        num_examples += len(curr_data)
        
    # Take the classification and re-make data
    new_data_with_class = np.array([[]])
    for classification, curr_data in enumerate(all_data):
        for train_example in curr_data:
            with_class = np.append(train_example, classification)            
            new_data_with_class = np.append(new_data_with_class, with_class)
    
    # Reshape data to proper format
    new_data_with_class = np.reshape(new_data_with_class, (num_examples, -1))
        
    np.random.shuffle(new_data_with_class)
        
    # X Shape: (num_examples, num_features)
    # Y Shape: (num_examples, 1)
    X, Y = np.split(new_data_with_class,[-1],axis=1)
    
    return X, Y
        
# Full logistic regression model output given just the data (optionally percent of data to train on)
def logistic_regression_full(all_data, train_percent = 0.8):
    print("augmenting data")
    try: 
        all_data = augment_data(all_data)
        print(f'AFTER AUGMENTATION\nclass 0: {len(all_data[0])}\nclass 10: {len(all_data[1])}')
    except: 
        print("error w/ augmentation")
        return None

    X, Y = x_y_data_create(all_data)
    
    # Calc number of examples for training and rest for testing
    num_examples = np.shape(X)[0]
    train_examples = int(train_percent * num_examples)
    
    # Split into train, test sets
    X_train = X[:train_examples]
    Y_train = Y[:train_examples]
    X_test = X[train_examples:]
    Y_test = Y[train_examples:]

    # Get the weights after 100 epochs of logistic regression training
    W, b = logistic_regression(X_train.T, Y_train.T, batch_size = train_examples, epochs = 100, learning_rate = 0.05)

    #print(W)
    #print(b)
    
    # Get predictions for logistic regression using these weights
    predictions_Y_hat, predictions_A = logistic_regression_prediction(X_test.T, Y_test.T, W, b)
    
    logistic_loss = NLL_loss_calc(predictions_A, Y_test.T)
    logistic_acc_metrics = accuracy_calc(predictions_Y_hat, Y_test.T)
    return logistic_acc_metrics