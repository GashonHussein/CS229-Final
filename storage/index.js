//main traverses a folder and uploads each file to the storage bucket
// ---- NAMING CONVENTION ----
// The name of the folder should be the classification of the photos it contains
// e.g 0
// assets are included in gitignore

const path = require("path");
const fs = require("fs");
const { upload } = require("./upload");

const main = (folder) => {
  const fileNames = fs.readdirSync(path.resolve(__dirname, `assets/${folder}`));
  fileNames.forEach((file) => {
    upload(`./assets/${folder}/${file}`, folder);
  });
};

// e.g
main("0");
