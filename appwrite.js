const sdk = require("node-appwrite");

const client = new sdk.Client();

function initializeAppWrite() {
  client
    .setEndpoint("https://fra.cloud.appwrite.io/v1")
    .setProject(process.env.APPWRITE_PROJECT_ID)
    .setKey(process.env.APPWRITE_KEY)
    .setSelfSigned();
}

module.exports = {
  initializeAppWrite,
  client,
};
