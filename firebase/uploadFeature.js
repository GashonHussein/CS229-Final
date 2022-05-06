var admin = require("firebase-admin");

var serviceAccount = require("./service/service-account.json");

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

const firestore = admin.firestore();


const uploadFeature = (queryName, arr) => {
  firestore.collection("rgb_features").doc(`${queryName}`).set({
    features: arr
  }).then(() => console.log("uploaded", queryName))
    .catch((err) => console.log("err", err))
}

uploadFeature("test", [10, 10, 10])

