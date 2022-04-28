import sys
sys.path.append('../storage')
import get_images
from get_images import get_images_url

# this function will is called for each image
def process_image(image_url):
    print(image_url)

def main():
    for i in range(0, 11, 5):
        res = get_images_url(i)
        for image in res:
            process_image(image)

if __name__ == "__main__":
    main()