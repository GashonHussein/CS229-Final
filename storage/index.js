//main traverses a folder and uploads each file to the storage bucket
// ---- NAMING CONVENTION ----
// The name of the folder should be the classification of the photos it contains
// e.g 0
// assets are included in gitignore

const path = require("path");
const fs = require("fs");
const { upload } = require("./upload");

const BATCH_SIZE = 150;

const main = async (folder) => {
  files = fs.readdirSync(path.resolve(__dirname, `assets/output/${folder}`));

  //This loop is responsible for uploading the images at a fast pace
  for (let i = 0; i < files.length / BATCH_SIZE; i++) {
    for (let j = 0; j < BATCH_SIZE; j++) {
      const index = i * BATCH_SIZE + j;
      const file = files[index];

      if (file != null) {
        fileName = file.split(".");
        console.log("\x1b[36m%s\x1b[0m", "batch", i, j);
        if (fileName[fileName.length - 1] == "png") {
          if (j == BATCH_SIZE - 1) {
            await upload(`./assets/output/${folder}/${file}`, folder);
          } else {
            upload(`./assets/output/${folder}/${file}`, folder).catch((e) => console.log(e))
          }
        }
      }
    }
  }
};

// e.g
(async () => {
  for (let i = 0; i <= 10; i += 5) {
    console.log(i);
    await main(`${i}`);
  }
})();
