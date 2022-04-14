import json

# this function returns a dict of image frames associated with their drowsiness classification
def get_images(classification):
    with open('./util/images.json', 'r') as f:
        data = json.load(f)
    return data["classification_{}".format(classification)]