// backend/controllers/analyzeController.js
import axios from "axios";
import fs from "fs";
import FormData from "form-data";
import { generateInsightsGemini } from "./geminiController.js";

const ML_ENGINE_URL = "http://127.0.0.1:5001/analyze";

export const analyzeAd = async (req, res) => {
  console.log("🟢 Route hit: /api/upload");

  try {
    // Validate file
    if (!req.file) {
      console.warn("❌ No file received");
      return res.status(400).json({ error: "No file uploaded" });
    }

    const filePath = req.file.path;
    console.log("📸 Received file:", filePath);

    // Step 1: Send to Python ML Engine
    const formData = new FormData();
    formData.append("file", fs.createReadStream(filePath));

    console.log("➡️ Sending to ML Engine...");
    const mlResponse = await axios.post(ML_ENGINE_URL, formData, {
      headers: formData.getHeaders(),
      timeout: 90000, // allow 90s for ML processing
    });

    const mlData = mlResponse.data;
    console.log("✅ ML Engine result:", mlData);

    // Step 2: Delete temp file
    try {
      fs.unlinkSync(filePath);
      console.log("🧹 Temp file deleted");
    } catch (err) {
      console.warn("⚠️ Could not delete file:", err.message);
    }

    // Step 3: Send enriched features to Gemini
    console.log("🤖 Sending features to Gemini...");
    const geminiResult = await generateInsightsGemini(mlData);
    console.log("✅ Gemini returned:", geminiResult);

    // Step 4: Merge and respond
    const finalResult = {
      ...mlData,
      ...geminiResult,
    };

    console.log("🚀 Final result to frontend:", finalResult);
    return res.status(200).json(finalResult);
  } catch (error) {
    console.error("🔥 Error in analyzeAd:", error.message);

    if (error.code === "ECONNABORTED") {
      return res.status(504).json({ error: "ML Engine request timed out" });
    }

    if (error.response) {
      console.error("↩️ ML Engine error:", error.response.data);
      return res.status(502).json({ error: error.response.data });
    }

    return res.status(500).json({ error: "Analysis failed" });
  }
};
