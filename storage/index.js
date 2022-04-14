if (process.env.NODE_ENV !== "production") {
  require("dotenv").config();
}
const fs = require("fs");

const { Storage } = require("@google-cloud/storage");
const storage = new Storage();
const bucket = storage.bucket(process.env.GCLOUD_STORAGE_BUCKET);

const images = require("./util/images.json");

const randomHash = () => Math.random().toString(36).substring(7);

//update images.json
const updateJSON = (images) => {
  const data = JSON.stringify(images, null, 2);

  fs.writeFile("./util/images.json", data, (err) => {
    if (err) throw err;
    console.log("Data written to file");
  });
};

const handleUpload = (data) => {
  const uri = `https://storage.googleapis.com/${process.env.GCLOUD_STORAGE_BUCKET}/${data.imgRef}`;

  images[`classification_${data.folder}`][data.imgRef] = { ...data, uri: uri };
  updateJSON(images);
};

const upload = (path, classification) => {
  const options = {
    destination: `${classification}/${randomHash()}.png`,
    public: true,
    resumable: true,
    metadata: {
      metadata: {
        classification: `${classification}`,
      },
    },
  };
  bucket
    .upload(path, options)
    .then((res) =>
      handleUpload({ folder: classification, imgRef: res[1].name, mediaRef: res[1].mediaLink })
    )
    .catch((err) => console.log("error", err));
  return;
};

module.exports = { upload };
