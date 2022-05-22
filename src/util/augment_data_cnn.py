import numpy as np

def flip_data_cnn(example):
    example_copy = np.copy(example)
    flipped_example = np.flip(example_copy, axis = 1)
    
    return flipped_example

def augment_data_cnn(all_data):
    class_0_amount = len(all_data[0])
    class_1_amount = len(all_data[1])
    
    modify_index = 0
    
    while modify_index < class_1_amount and len(all_data[1]) < class_0_amount:
        curr_example = all_data[1][modify_index]
        new_example = flip_data_cnn(curr_example)
        
        all_data[1].append(new_example)
        
        modify_index += 1
    
    new_class_1_amount = len(all_data[1])
    class_amounts_diff = class_0_amount - new_class_1_amount
    
    if class_amounts_diff <= 0:
        return all_data
    
    rand_indexes_list = np.random.randint(new_class_1_amount, size = class_amounts_diff)
        
    for index in rand_indexes_list:
        example_copy = np.copy(all_data[1][index])
        all_data[1].append(example_copy)
        
    return all_data
