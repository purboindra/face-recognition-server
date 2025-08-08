const { faceapi } = require("./models");
let labeled = null;
let matcher = null;
let lastVersion = 0;
let currentThreshold = 0.55;

function buildMatcher(documents, distance = 0.55) {
  labeled = documents.map(
    (u) =>
      new faceapi.LabeledFaceDescriptors(u.userName, [
        new Float32Array(u.descriptor),
      ])
  );
  matcher = new faceapi.FaceMatcher(labeled, distance);
}

async function ensure(databases, DB_ID, COL_ID, threshold = 0.55, listAllFn) {
  if (matcher || building) return;
  building = true;
  try {
    const docs = await listAllFn(databases, DB_ID, COL_ID);
    await rebuildFromDocuments(docs, threshold);
  } finally {
    building = false;
  }
}

async function rebuildFromDocuments(documents, threshold = currentThreshold) {
  const labeled = documents
    .filter((d) => Array.isArray(d.descriptor) && d.descriptor.length > 0)
    .map(
      (d) =>
        new faceapi.LabeledFaceDescriptors(d.userName, [
          new Float32Array(d.descriptor),
        ])
    );

  matcher = new faceapi.FaceMatcher(labeled, threshold);
  currentThreshold = threshold;
  lastBuiltAt = Date.now();
}

module.exports = {
  getMatcher() {
    return matcher;
  },
  rebuildMatcher(documents, distance) {
    buildMatcher(documents, distance);
    lastVersion++;
  },
  bumpVersion() {
    lastVersion++;
  },
  getVersion() {
    return lastVersion;
  },
};
