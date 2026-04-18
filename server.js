// backend/server.js
import * as dotenv from "dotenv";
dotenv.config(); // ✅ Load environment variables BEFORE any other imports

import express from "express";
import cors from "cors";
import morgan from "morgan";
import path from "path";
import { fileURLToPath } from "url";
import analyzeRoute from "./routes/analyze.js";

// ------------------------------------------------------------
// 📦 Environment Setup
// ------------------------------------------------------------
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PORT = process.env.PORT || 5050;

const app = express();

// ------------------------------------------------------------
// 🧩 Middleware
// ------------------------------------------------------------
app.use(
  cors({
    origin: "*", // TODO: restrict in production
    methods: ["GET", "POST"],
    allowedHeaders: ["Content-Type"],
  })
);
app.use(express.json());
app.use(morgan("dev"));

// ------------------------------------------------------------
// 🖼️ Static File Serving (ML output & previews)
// ------------------------------------------------------------
app.use("/outputs", express.static(path.join(__dirname, "../MLEngine")));

// ------------------------------------------------------------
// 🧠 Health & Diagnostic Routes
// ------------------------------------------------------------
app.get("/ping", (_, res) => res.json({ message: "Backend API working fine" }));
app.get("/hello", (_, res) => res.json({ message: "Hello from AdMind backend" }));
app.get("/test", (_, res) => {
  console.log("✅ /test route hit");
  res.json({ message: "Express route working fine" });
});

// ------------------------------------------------------------
// 🔗 API Routes
// ------------------------------------------------------------
app.use("/api", analyzeRoute);

// ------------------------------------------------------------
// ⚠️ Error Handling Middleware
// ------------------------------------------------------------
app.use((err, req, res, _next) => {
  console.error("🔥 Server Error:", err.stack);
  res.status(500).json({ error: "Internal server error" });
});

// ------------------------------------------------------------
// 🚀 Start Server
// ------------------------------------------------------------
app.listen(PORT, () => {
  console.log(`🚀 AdMind Backend running at http://127.0.0.1:${PORT}`);
  console.log(`🌐 Serving ML output images from /outputs`);
});
