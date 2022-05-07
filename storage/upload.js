if (process.env.NODE_ENV !== "production") {
  require("dotenv").config();
}
const fs = require("fs");

const { Storage } = require("@google-cloud/storage");
const storage = new Storage({ keyFilename: "./credentials/config.json" });
const bucket = storage.bucket(process.env.GCLOUD_STORAGE_BUCKET || "cs-229-storage-bucket");

const OUTPUT_FOLDER = "reduced"

const images = require("./util/images.json");
let imageList = images;

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
  const uri = `https://storage.googleapis.com/${process.env.GCLOUD_STORAGE_BUCKET || "cs-229-storage-bucket"}/${data.img_ref}`;

  console.log({ ...data, uri: uri });
  if (imageList[OUTPUT_FOLDER] == null) {
    imageList[OUTPUT_FOLDER] = {};
  }
  if (imageList[OUTPUT_FOLDER][`classification_${data.folder}`] == null) {
    imageList[OUTPUT_FOLDER][`classification_${data.folder}`] = {};
  }
  imageList[OUTPUT_FOLDER][`classification_${data.folder}`][data.img_ref] = { ...data, uri: uri };
  updateJSON(imageList);
};

const upload = async (path, classification) => {
  const options = {
    destination: `drowsiness/${OUTPUT_FOLDER}/${classification}/${randomHash()}.png`,
    public: true,
    resumable: true,
    metadata: {
      metadata: {
        classification: `${classification}`,
      },
    },
  };
  console.log("uploading", path);
  const res = await bucket.upload(path, options);
  handleUpload({ folder: classification, img_ref: res[1].name, media_ref: res[1].mediaLink });
  return;
};

module.exports = { upload };
