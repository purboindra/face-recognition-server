var express = require("express");
var router = express.Router();
var multer = require("multer");

var path = require("path");

const faceapi = require("face-api.js");
const canvas = require("canvas");
const { client } = require("../appwrite");
const { InputFile } = require("node-appwrite/file");
const { Databases, Query, ID, Storage } = require("node-appwrite");
const {
  APPWRITE_USER_DATABASE_ID,
  APPWRITE_USER_COLLECTION_ID,
  APPWRITE_BUCKER_ID,
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

  const MODEL_PATH = path.join(__dirname, "../public/models");

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
      message: "Image is required",
    });
    return;
  }

  const { username, longitude, latitude } = req.body;

  if (!username) {
    res.status(400).json({
      message: "Username is required",
    });
    return;
  }

  if (!longitude || !latitude) {
    res.status(400).json({
      message: "Location is required",
    });
    return;
  }

  const user = await databases.listDocuments(
    APPWRITE_USER_DATABASE_ID,
    APPWRITE_USER_COLLECTION_ID,
    [Query.equal("userName", [username])]
  );

  console.log(`User documents: ${user.documents}`);

  if (user.documents.length > 0) {
    res.status(400).json({
      message: "Username already registered",
    });
    return;
  }

  try {
    const img = await loadImage(file.path);

    const detections = await faceapi.detectAllFaces(
      img,
      new faceapi.TinyFaceDetectorOptions()
    );

    if (detections.length !== 1) {
      res.status(400).json({
        message: "Please ensure only your face is in the frame",
        code: "ERROR_MANY_FACES",
      });
      return;
    }

    const resultDescriptor = await faceapi
      .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!resultDescriptor) {
      res.status(400).json({
        message: "No face detected",
        code: "DETECTING_DESCRIPTOR_ERROR",
      });
      return;
    }

    const landmarks = resultDescriptor.landmarks;

    const leftEye = landmarks.getLeftEye();
    const rightEye = landmarks.getRightEye();

    if (!leftEye || !rightEye || Math.abs(leftEye[0].x - rightEye[0].x) < 10) {
      res.status(400).json({
        message: "Please face the camera directly (frontal view required)",
        code: "ERROR_NO_FACE",
      });
      return;
    }

    const box = resultDescriptor.detection.box;
    const imageCenterX = img.width / 2;
    const faceCenterX = box.x + box.width / 2;

    const offsetX = Math.abs(imageCenterX - faceCenterX);

    if (offsetX > img.width * 0.2) {
      return res.status(400).json({
        message: "Please center your face in the frame",
        code: "ERROR_FACE_NOT_CENTERED",
      });
    }

    const canvasInstance = canvas.createCanvas(img.width, img.height);
    const ctx = canvasInstance.getContext("2d");
    ctx.drawImage(img, 0, 0, img.width, img.height);
    const imageData = ctx.getImageData(0, 0, img.width, img.height).data;

    let totalBrightness = 0;
    for (let i = 0; i < imageData.length; i += 4) {
      const r = imageData[i];
      const g = imageData[i + 1];
      const b = imageData[i + 2];
      const brightness = (r + g + b) / 3;
      totalBrightness += brightness;
    }

    const averageBrightness = totalBrightness / (imageData.length / 4);

    if (averageBrightness < 40) {
      return res.status(400).json({
        message: "Image too dark, please ensure better lighting",
        code: "ERROR_LOW_LIGHT",
      });
    }

    const descriptor = Array.from(resultDescriptor.descriptor);

    const notCurrentUsers = await databases.listDocuments(
      APPWRITE_USER_DATABASE_ID,
      APPWRITE_USER_COLLECTION_ID,
      [Query.notEqual("userName", [username])]
    );

    const allUsers = notCurrentUsers.documents;

    if (allUsers.length > 0) {
      const labeledDescriptors = allUsers.map((user) => {
        const desc = JSON.parse(user.descriptor);
        return new faceapi.LabeledFaceDescriptors(user.userName, [
          new Float32Array(desc),
        ]);
      });

      const matcher = new faceapi.FaceMatcher(labeledDescriptors, 0.5);

      const bestMatch = matcher.findBestMatch(descriptor);

      if (bestMatch.label !== "unknown") {
        res.status(400).json({
          message: `This face already exists and is registered with ${bestMatch.label}`,
          code: "DUPLICATE_FACE",
        });
        return;
      }
    }

    const storage = new Storage(client);

    const nodeFile = InputFile.fromPath(file.path, `${username}.jpg`);

    const responseFile = await storage.createFile(
      APPWRITE_BUCKER_ID,
      ID.unique(),
      nodeFile
    );

    const imageUrl = `https://fra.cloud.appwrite.io/v1/storage/buckets/${APPWRITE_BUCKER_ID}/files/${responseFile.$id}/view?project=${APPWRITE_PROJECT_ID}`;

    await databases.createDocument(
      APPWRITE_USER_DATABASE_ID,
      APPWRITE_USER_COLLECTION_ID,
      ID.unique(),
      {
        userName: username,
        descriptor: descriptor,
        imageUrl: imageUrl,
        longitude: longitude,
        latitude: latitude,
      }
    );

    res.status(200).json({
      message: "Face registered successfully",
    });
  } catch (error) {
    console.error("Error register face", error);
    res.status(500).json({
      message: "Error register face",
    });
    return;
  } finally {
    fs.unlinkSync(file.path);
  }
});

module.exports = router;
