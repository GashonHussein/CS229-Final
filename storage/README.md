# Storage

## Dev Abstraction (get_images.py)

### get_images_url() (recommended)

`get_images_url(classification)` returns an array of img_ref urls associated with the drowsiness classification argument

### get_images()

`get_images(classification)` returns more info than `get_images_url()`. Returns a dict (url, media_ref, and folder) of image frames associated with their drowsiness classification

## General

- Linked to GCP Storage Bucket: find credentials in credentials/config.json
- Uploads an image to the bucket
  - the image is stored in a folder that identifies its classification
  - the image is named through a random hash
    - e.g. 6/headghae.png where 6 is the classification & folder of the image
  - information about the upload is stored in util/images.json
    - includes uri link, mediaRef (for download), folder, and name
