require("dotenv").config();

var express = require("express");
var router = express.Router();
var multer = require("multer");

var path = require("path");

const fs = require("fs");

const faceapi = require("face-api.js");
const canvas = require("canvas");

const { client } = require("../appwrite");
const { Databases, Query, ID, Storage } = require("node-appwrite");
const {
  APPWRITE_USER_DATABASE_ID,
  APPWRITE_USER_COLLECTION_ID,
} = require("./constants");

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "public/uploads");
  },

  filename: function (req, file, cb) {
    const ext = path.extname(file.originalname);
    cb(null, `${Date.now()}${ext}`);
  },
});

var upload = multer({ storage: storage });

router.post("/", upload.single("image"), async (req, res) => {
  const databases = new Databases(client);

  const file = req.file;

  const MODEL_PATH = path.join(__dirname, "../../public/models");

  console.log(`MODEL_PATH: ${MODEL_PATH}`);
  console.log(`DIR NAME: ${__dirname}`);

  const { Canvas, Image, ImageData, loadImage } = canvas;
  faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

  await faceapi.nets.ssdMobilenetv1.loadFromDisk(
    `${MODEL_PATH}/ssd_mobilenetv1`
  );
  await faceapi.nets.faceLandmark68Net.loadFromDisk(
    `${MODEL_PATH}/face_landmark_68`
  );
  await faceapi.nets.faceRecognitionNet.loadFromDisk(
    `${MODEL_PATH}/face_recognition`
  );
  await faceapi.nets.tinyFaceDetector.loadFromDisk(
    `${MODEL_PATH}/tiny_face_detector`
  );

  if (!file) {
    res.status(400).json({
      message: "Tidak ada wajah yang terdeteksi",
    });
    return;
  }

  const { username } = req.body;

  if (!username) {
    res.status(400).json({
      message: "Username diperlukan",
    });
    return;
  }

  try {
    const users = await databases.listDocuments(
      APPWRITE_USER_DATABASE_ID,
      APPWRITE_USER_COLLECTION_ID
    );

    const labeledDescriptors = users.documents.map((user) => {
      return new faceapi.LabeledFaceDescriptors(user.userName, [
        new Float32Array(JSON.parse(user.descriptor)),
      ]);
    });

    const user = await databases.listDocuments(
      APPWRITE_USER_DATABASE_ID,
      APPWRITE_USER_COLLECTION_ID,
      [Query.equal("userName", [username])]
    );

    if (user.total === 0) {
      res.status(400).json({
        message: "Username ini belum terdaftar",
        code: "USER_NOT_FOUND",
      });
      return;
    }

    const img = await loadImage(file.path);

    const singleResult = await faceapi
      .detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!singleResult) {
      res.status(400).json({
        message: "Tidak dapat mendeteksi wajah",
        code: "DETECTING_DESCRIPTOR_ERROR",
      });
      return;
    }

    const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.5);

    const bestMatch = faceMatcher.findBestMatch(singleResult.descriptor);

    if (bestMatch.label === "unknown") {
      res.status(401).json({
        matched: false,
        label: null,
        distance: bestMatch.distance,
        message: "Wajah ini belum terdaftar",
      });
      return;
    }

    if (bestMatch.label !== username) {
      res.status(400).json({
        matched: false,
        identifiedAs: bestMatch.label,
        distance: bestMatch.distance,
        message: `Wajah ini terdaftar sebagai ${bestMatch.label}`,
      });
      return;
    }

    res.status(200).json({
      matched: true,
      identifiedAs: bestMatch.label,
      distance: bestMatch.distance,
      message: "Verifikasi wajah berhasil",
    });
  } catch (error) {
    console.error("Error register face", error);
    res.status(500).json({
      message: "Terjadi kesalahan saat memverifikasi wajah",
    });
    return;
  } finally {
    fs.unlinkSync(file.path);
  }
});

module.exports = router;
