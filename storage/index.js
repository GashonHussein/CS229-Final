//main traverses a folder and uploads each file to the storage bucket
// ---- NAMING CONVENTION ----
// The name of the folder should be the classification of the photos it contains
// e.g 0
// assets are included in gitignore

const path = require("path");
const fs = require("fs");
const { upload } = require("./upload");

const main = (folder) => {
  fs.readdir(path.resolve(__dirname, `assets/${folder}`), (err, files) => {
    files.forEach((file) => {
      upload(`./assets/${folder}/${file}`, folder);
    });
  });
};

// e.g
(async () => {
  for (let i = 0; i <= 10; i++) {
    console.log(i);
    await main(`${i}`);
  }
})();
// main("1");
