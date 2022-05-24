import numpy as np


def flip_data(example):
    copy_example = np.copy(example)
    pixels = np.shape(example)[1]/3
    
    assert (pixels ** 0.5)%1 == 0
    
    rows = int(pixels ** 0.5)
    cols = int(rows)
    
    reshaped_example = np.reshape(copy_example, (rows, cols, 3))
    
    flipped_reshaped_example = np.flip(reshaped_example, axis = 1)
        
    return flipped_reshaped_example.reshape(1, -1)
    

def augment_data(all_data):
    class_0_amount = len(all_data[0])
    class_1_amount = len(all_data[1])
    print(f'class 0: {class_0_amount}\nclass 10: {class_1_amount}')
    
    modify_index = 0
    
    while modify_index < class_1_amount and len(all_data[1]) < class_0_amount:
        curr_example = all_data[1][modify_index]
        new_example = flip_data(curr_example)
        
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
