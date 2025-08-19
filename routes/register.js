require("dotenv").config();

var express = require("express");
var router = express.Router();
var multer = require("multer");
const sharp = require("sharp");

var fs = require("fs");

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
  APPWRITE_PROJECT_ID,
} = require("./constants");
const { loadModels } = require("../models");

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "public/uploads");
  },

  filename: function (req, file, cb) {
    const ext = path.extname(file.originalname);
    cb(null, `${Date.now()}${ext}`);
  },
});

var upload = multer({
  storage: storage,
  limits: { fileSize: 4 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    if (!/^image\/(jpe?g|png|webp)$/i.test(file.mimetype))
      return cb(new Error("Invalid file type"));
    cb(null, true);
  },
});

async function detectSingleRobust(img, opts) {
  let d = await faceapi
    .detectSingleFace(img, opts)
    .withFaceLandmarks()
    .withFaceDescriptor();
  if (d) return d;

  d = await faceapi
    .detectSingleFace(img, opts)
    .withFaceLandmarks()
    .withFaceDescriptor();
  if (d) return d;

  const ssdOpts = new faceapi.SsdMobilenetv1Options({ minConfidence: 0.2 });
  d = await faceapi
    .detectSingleFace(img, ssdOpts)
    .withFaceLandmarks()
    .withFaceDescriptor();

  return d || null;
}

async function listAllOthers(databases, DB, COL, username) {
  const out = [];
  let cursor = null;
  while (true) {
    const q = [Query.notEqual("userName", [username]), Query.limit(100)];
    if (cursor) q.push(Query.cursorAfter(cursor));
    const page = await databases.listDocuments(DB, COL, q);
    out.push(...page.documents);
    if (page.documents.length < 100) break;
    cursor = page.documents[page.documents.length - 1].$id;
  }
  return out;
}

function eyeCenter(eye) {
  const cx = eye.reduce((s, p) => s + p.x, 0) / eye.length;
  const cy = eye.reduce((s, p) => s + p.y, 0) / eye.length;
  return { cx, cy };
}

function meanPoint(points) {
  const n = points.length;
  const sx = points.reduce((s, p) => s + p.x, 0);
  const sy = points.reduce((s, p) => s + p.y, 0);
  return { x: sx / n, y: sy / n };
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function checkPitch(landmarks, { low = 0.38, high = 0.62 } = {}) {
  const leftEyeC = meanPoint(landmarks.getLeftEye());
  const rightEyeC = meanPoint(landmarks.getRightEye());
  const eyeLine = {
    x: (leftEyeC.x + rightEyeC.x) / 2,
    y: (leftEyeC.y + rightEyeC.y) / 2,
  };

  const mouthC = meanPoint(landmarks.getMouth());
  const jaw = landmarks.getJawOutline();
  const chin = jaw[8];

  const eyeToMouth = dist(eyeLine, mouthC);
  const eyeToChin = dist(eyeLine, chin);

  const pitchRatio = eyeToMouth / (eyeToMouth + eyeToChin);
  const ok = pitchRatio >= low && pitchRatio <= high;

  return {
    ok,
    pitchRatio,
    reason: "ERROR_FACE_UP_DOWN",
    message: "Wajah terdeteksi tidak menghadap ke kamera",
  };
}

function checkYaw(landmarks, { threshold = 0.18 } = {}) {
  const leftEyeC = meanPoint(landmarks.getLeftEye());
  const rightEyeC = meanPoint(landmarks.getRightEye());
  const noseC = meanPoint(landmarks.getNose());

  const dL = dist(noseC, leftEyeC);
  const dR = dist(noseC, rightEyeC);

  const yawScore = Math.log(dR / dL);

  return {
    ok: Math.abs(yawScore) <= threshold,
    yawScore,
    reason: "ERROR_FACE_TURNED",
    message: "Wajah terdeteksi tidak menghadap ke kamera",
  };
}

function checkPoseAndCenter(landmarks, box, imgW) {
  const L = eyeCenter(landmarks.getLeftEye());
  const R = eyeCenter(landmarks.getRightEye());
  const roll = (Math.atan2(R.cy - L.cy, R.cx - L.cx) * 180) / Math.PI;
  if (Math.abs(roll) > 20)
    return {
      ok: false,
      reason: "ERROR_FACE_TILTED",
      message: "Wajah terdeteksi tidak menghadap ke kamera",
    };

  const yaw = checkYaw(landmarks, { threshold: 0.18 });

  if (!yaw.ok) {
    return {
      ok: false,
      message: yaw.message,
      reason: yaw.reason,
      yawScore: yaw.yawScore,
    };
  }

  // const pitch = checkPitch(landmarks, { low: 0.38, high: 0.62 });

  // if (!pitch.ok) {
  //   return {
  //     ok: false,
  //     message: pitch.message,
  //     reason: pitch.reason,
  //     pitchRatio: pitch.pitchRatio,
  //   };
  // }

  const imageCenterX = imgW / 2;
  const faceCenterX = box.x + box.width / 2;
  const offsetX = Math.abs(imageCenterX - faceCenterX);

  if (offsetX > imgW * 0.2)
    return {
      ok: false,
      message: "Wajah terdeteksi tidak berada di tengah frame",
      reason: "ERROR_FACE_NOT_CENTERED",
    };

  const eyeDist = Math.hypot(L.cx - R.cx, L.cy - R.cy);
  if (eyeDist < box.width * 0.25)
    return {
      ok: false,
      message: "Wajah terdeteksi terlalu kecil",
      reason: "ERROR_FACE_TOO_SMALL",
    };

  return { ok: true };
}

function varianceOfLaplacian(ctx, w, h) {
  const src = ctx.getImageData(0, 0, w, h).data;
  return 150;
}

function avgLumaInBox(ctx, box) {
  const x = Math.max(0, Math.floor(box.x));
  const y = Math.max(0, Math.floor(box.y));
  const w = Math.max(1, Math.floor(box.width));
  const h = Math.max(1, Math.floor(box.height));

  const { data } = ctx.getImageData(x, y, w, h);
  let sum = 0,
    n = 0;

  const stride = 2;
  for (let row = 0; row < h; row += stride) {
    for (let col = 0; col < w; col += stride) {
      const i = (row * w + col) * 4;
      const r = data[i],
        g = data[i + 1],
        b = data[i + 2];
      const yLuma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      sum += yLuma;
      n++;
    }
  }
  return sum / Math.max(1, n);
}

function splitBox(box) {
  const halfW = Math.floor(box.width / 2);
  const left = { x: box.x, y: box.y, width: halfW, height: box.height };
  const right = {
    x: box.x + halfW,
    y: box.y,
    width: box.width - halfW,
    height: box.height,
  };
  return { left, right };
}

router.post("/", upload.single("image"), async (req, res) => {
  await loadModels();

  const databases = new Databases(client);

  if (!req.file)
    return res.status(400).json({ message: "File gambar diperlukan" });

  const resizedPath = req.file.path.replace(/(\.\w+)$/, "-resized$1");

  await sharp(req.file.path)
    .resize({ width: 1280, height: 1280, fit: "inside" })
    .toColorspace("srgb")
    .jpeg({ quality: 92, mozjpeg: true })
    .toFile(resizedPath);

  const opts = new faceapi.TinyFaceDetectorOptions({
    inputSize: 608,
    scoreThreshold: 0.5,
  });

  const usernameRaw = (req.body.username || "").trim();
  const username = usernameRaw.toLowerCase();
  const longitude = Number(req.body.longitude);
  const latitude = Number(req.body.latitude);

  if (!username)
    return res.status(400).json({ message: "Username diperlukan" });
  if (!Number.isFinite(longitude) || !Number.isFinite(latitude)) {
    return res.status(400).json({ message: "Lokasi tidak valid" });
  }

  try {
    const existing = await databases.listDocuments(
      APPWRITE_USER_DATABASE_ID,
      APPWRITE_USER_COLLECTION_ID,
      [Query.equal("userName", [usernameRaw]), Query.limit(1)]
    );

    if (existing.total > 0) {
      return res.status(400).json({ message: "Username ini sudah terdaftar" });
    }

    const img = await canvas.loadImage(resizedPath);

    const canvasInstance = canvas.createCanvas(img.width, img.height);
    const ctx = canvasInstance.getContext("2d");
    ctx.drawImage(img, 0, 0, img.width, img.height);

    const blurScore = varianceOfLaplacian(ctx, img.width, img.height);
    if (blurScore < 120) {
      return res.status(400).json({
        message: "Foto terlalu buram, silakan ambil ulang",
        code: "ERROR_BLUR",
      });
    }

    console.log("tiny loaded?", faceapi.nets.tinyFaceDetector.isLoaded);
    console.log("landmark loaded?", faceapi.nets.faceLandmark68Net.isLoaded);
    console.log("recog loaded?", faceapi.nets.faceRecognitionNet.isLoaded);
    console.log(
      "ssd mobilenetv1 loaded?",
      faceapi.nets.ssdMobilenetv1.isLoaded
    );

    const single = await detectSingleRobust(img, opts);

    if (!single) {
      return res
        .status(400)
        .json({ message: "Tidak dapat mendeteksi wajah", code: "NO_FACE" });
    }

    const faceBox = single.detection.box;
    const avgLuma = avgLumaInBox(ctx, faceBox);

    const { left, right } = splitBox(faceBox);
    const lumaL = avgLumaInBox(ctx, left);
    const lumaR = avgLumaInBox(ctx, right);
    const diff = Math.abs(lumaL - lumaR);

    console.log("avgLuma(face):", Math.round(avgLuma));
    console.log("lumaL:", Math.round(lumaL));
    console.log("lumaR:", Math.round(lumaR));
    console.log("diff:", diff);

    if (lumaL < 50 && lumaR < 50 && diff > 18) {
      return res.status(400).json({
        message:
          "Pencahayaan tidak merata pada wajah, harap cari tempat lebih terang",
        code: "ERROR_LOW_LIGHT",
        meta: { lumaL: Math.round(lumaL), lumaR: Math.round(lumaR) },
      });
    }
    const {
      detection: { box },
      landmarks,
    } = single;

    const descriptor = Array.from(single.descriptor);

    // const others = await listAllOthers(
    //   databases,
    //   APPWRITE_USER_DATABASE_ID,
    //   APPWRITE_USER_COLLECTION_ID,
    //   username
    // );

    const others = await databases.listDocuments(
      APPWRITE_USER_DATABASE_ID,
      APPWRITE_USER_COLLECTION_ID,
      [Query.notEqual("userName", [username]), Query.limit(100)]
    );

    if (others.total > 0) {
      const labeled = others.documents.map(
        (u) =>
          new faceapi.LabeledFaceDescriptors(u.userName, [
            new Float32Array(u.descriptor),
          ])
      );
      const matcher = new faceapi.FaceMatcher(labeled, 0.55);
      const best = matcher.findBestMatch(new Float32Array(descriptor));

      if (best.label !== "unknown") {
        return res.status(400).json({
          message: `Wajah ini sudah terdaftar sebagai ${best.label}`,
          code: "DUPLICATE_FACE",
        });
      }
    }

    const { ok, reason, message } = checkPoseAndCenter(
      landmarks,
      box,
      img.width
    );

    if (!ok) {
      return res
        .status(400)
        .json({ message: message || "UNKNOWN", code: reason || "UNKNOWN" });
    }

    const appwriteStorage = new Storage(client);
    const nodeFile = InputFile.fromPath(req.file.path, `${username}.jpg`);
    const uploaded = await appwriteStorage.createFile(
      APPWRITE_BUCKER_ID,
      ID.unique(),
      nodeFile
    );
    const imageUrl = `https://fra.cloud.appwrite.io/v1/storage/buckets/${APPWRITE_BUCKER_ID}/files/${uploaded.$id}/view?project=${APPWRITE_PROJECT_ID}`;

    await databases.createDocument(
      APPWRITE_USER_DATABASE_ID,
      APPWRITE_USER_COLLECTION_ID,
      ID.unique(),
      {
        userName: usernameRaw,
        // userNameNormalized: username,
        descriptor,
        imageUrl,
        longitude: longitude.toString(),
        latitude: latitude.toString(),
      }
    );

    return res
      .status(200)
      .json({ message: `Wajah berhasil terdaftar sebagai ${usernameRaw}` });
  } catch (err) {
    console.error(err);
    return res
      .status(500)
      .json({ message: "Terjadi kesalahan saat mendaftarkan wajah" });
  } finally {
    if (req.file) {
      try {
        await fs.promises.unlink(resizedPath);
        if (req.file) {
          await fs.promises.unlink(req.file.path);
        }
      } catch (_) {}
    }
  }
});

module.exports = router;
