# Storage

### Init

`npm i`

`node main`

## General

- Linked to GCP Storage Bucket: find credentials in credentials/config.json
- Uploads an image to the bucket
  - the image is stored in a folder that identifies its classification
  - the image is named through a random hash
    - e.g. 6/headghae.png where 6 is the classification of the image
  - information about the upload is stored in util/images.json
    - includes uri link, mediaRef (for download), folder, and name
