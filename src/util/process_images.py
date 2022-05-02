import sys
import skimage.io as io

sys.path.append('../storage')
from get_images import get_images_url


# this function is called for each image in the database
def process_image(image_url):
    # # Read
    # stackColors = io.imread('single_color_test_image.png')
    # # Split
    # red_features_matrix = stackColors[:, :, 0]
    # green_features_matrix = stackColors[:, :, 1]
    # blue_features_matrix = stackColors[:, :, 2]
    # # For CNN Keep channels in place
    # cnn_features_stack = stackColors[:, :, :3]
    # # For other models (logistic regression) flatten features/channels
    # flattened_features = cnn_features_stack.reshape(1, -1)

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


def main():
    for i in range(0, 11, 5):
        res = get_images_url(i)
        for image in res:
            process_image(image)


if __name__ == "__main__":
    main()
