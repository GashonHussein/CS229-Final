from get_images import get_images_url
import sys

sys.path.append('../storage')


# this function is called for each image in the database
def process_image(image_url):
    print(image_url)


def main():
    print("executing")
    for i in range(0, 11, 5):
        res = get_images_url(i)
        for image in res:
            process_image(image)


if __name__ == "__main__":
    main()
