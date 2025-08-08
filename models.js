const path = require("path");
const faceapi = require("face-api.js");
const canvas = require("canvas");
const { Canvas, Image, ImageData } = canvas;

faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

let loaded = false;
async function loadModels() {
  if (loaded) return;
  const MODEL_PATH = path.join(__dirname, "public/models");
  await Promise.all([
    faceapi.nets.faceLandmark68Net.loadFromDisk(
      `${MODEL_PATH}/face_landmark_68`
    ),
    faceapi.nets.faceRecognitionNet.loadFromDisk(
      `${MODEL_PATH}/face_recognition`
    ),
    faceapi.nets.tinyFaceDetector.loadFromDisk(
      `${MODEL_PATH}/tiny_face_detector`
    ),
    faceapi.nets.ssdMobilenetv1.loadFromDisk(`${MODEL_PATH}/ssd_mobilenetv1`),
  ]);
  loaded = true;
}
module.exports = { faceapi, canvas, loadModels };
