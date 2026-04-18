// backend/routes/analyze.js
import express from "express";
import multer from "multer";
import fs from "fs";
import path from "path";
import axios from "axios";
import FormData from "form-data";

import * as dotenv from "dotenv";
dotenv.config();

// ✅ Official Gemini SDK (new)
import { GoogleGenAI } from "@google/genai";

const router = express.Router();

/* -----------------------------------------------------------
 * Config
 * --------------------------------------------------------- */
const UPLOADS_DIR = path.join(process.cwd(), "uploads");
const MAX_UPLOAD_MB = Number(process.env.MAX_UPLOAD_MB || 200);
const ML_BASE = process.env.ML_BASE || "http://127.0.0.1:5001";

const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";
const PREFERRED_MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash";

// Fallback chain — gemini-1.5-flash removed (404 on v1beta), lite kept as backup
const MODEL_CANDIDATES = [
  PREFERRED_MODEL,
  "gemini-2.5-flash-lite",
  "gemini-2.5-flash-8b",
].filter(Boolean);

if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR, { recursive: true });

/* -----------------------------------------------------------
 * Multer
 * --------------------------------------------------------- */
const storage = multer.diskStorage({
  destination: (_, __, cb) => cb(null, UPLOADS_DIR),
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname || "").toLowerCase() || ".bin";
    const base = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    cb(null, `${base}${ext}`);
  },
});
const upload = multer({
  storage,
  limits: { fileSize: MAX_UPLOAD_MB * 1024 * 1024 },
});

/* -----------------------------------------------------------
 * Helpers
 * --------------------------------------------------------- */
const VIDEO_EXTS = new Set([".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"]);
const IMAGE_EXTS = new Set([".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".heic"]);

function looksLikeVideo({ mimetype = "", filename = "" }) {
  const mt = (mimetype || "").toLowerCase();
  if (mt.startsWith("video/")) return true;
  const ext = path.extname(filename || "").toLowerCase();
  return VIDEO_EXTS.has(ext);
}
function looksLikeImage({ mimetype = "", filename = "" }) {
  const mt = (mimetype || "").toLowerCase();
  if (mt.startsWith("image/")) return true;
  const ext = path.extname(filename || "").toLowerCase();
  return IMAGE_EXTS.has(ext);
}
function pickMlEndpoint({ isVideo, forceType }) {
  if (forceType === "video") return `${ML_BASE}/analyze_video`;
  if (forceType === "image") return `${ML_BASE}/analyze`;
  return isVideo ? `${ML_BASE}/analyze_video` : `${ML_BASE}/analyze`;
}

/* -----------------------------------------------------------
 * Gemini (new SDK) + robust extraction
 * --------------------------------------------------------- */
function makeGemini() {
  return new GoogleGenAI({ apiKey: GEMINI_API_KEY });
}

function extractGenAIText(resp) {
  // Primary shapes from @google/genai
  if (resp && typeof resp.text === "string") return resp.text.trim();
  if (resp && typeof resp.output_text === "string") return resp.output_text.trim();

  // Fallback to candidates/parts
  const parts =
    resp?.candidates?.[0]?.content?.parts ||
    resp?.output?.[0]?.content?.parts ||
    [];
  if (Array.isArray(parts) && parts.length) {
    const s = parts.map(p => (p?.text || "")).join("").trim();
    if (s) return s;
  }
  return "";
}

/** Keep only high-signal fields so prompts stay small and never overflow */
function compactContext(ml) {
  const media = ml?.media_type || "image";
  const base = {
    media_type: media,
    layout_balance: ml?.layout_balance ?? ml?.layout_balance_avg ?? null,
    creative_score: ml?.creative_score ?? null,
    nsfw_safe: ml?.nsfw?.is_safe ?? true,
  };

  if (media === "video") {
    const ve = ml?.video_emotions?.summary || {};
    return {
      ...base,
      final_emotion: ve.final_top ?? null,
      avg_faces_per_sec: ve.avg_faces_per_sec ?? null,
      objects_top: (ml?.objects_top || []).slice(0, 5),
      brands: (ml?.brands || []).slice(0, 8),
      ocr_excerpt: (ml?.ocr_excerpt || "").slice(0, 280),
      color_palette_global: (ml?.color_palette_global || []).slice(0, 6),
      duration_sec: ml?.duration_sec ?? null,
    };
  }

  // image
  return {
    ...base,
    dominant_emotion: ml?.dominant_emotion ?? null,
    emotion_confidence: ml?.emotion_confidence ?? null,
    face_count: ml?.face_count ?? null,
    detected_objects: (ml?.detected_objects || []).slice(0, 8),
    brands: (ml?.brands || []).slice(0, 8),
    text_content: (ml?.text_content || "").slice(0, 280),
    color_palette: (ml?.color_palette || []).slice(0, 6),
    top_categories: (ml?.top_categories || []).slice(0, 3),
  };
}

/** last-resort local heuristic so you ALWAYS get something */
function fallbackHeuristicInsight(ml) {
  const media = ml?.media_type || "image";
  const emotion =
    media === "video"
      ? ml?.video_emotions?.summary?.final_top || "neutral"
      : ml?.dominant_emotion || "neutral";

  const hasText =
    (media === "video" ? (ml?.ocr_excerpt || "") : (ml?.text_content || "")).trim().length > 0;

  const weakness = hasText ? "Headline hierarchy may be unclear." : "Lack of clear text/CTA.";
  const suggestions = [
    hasText ? "Clarify headline and CTA hierarchy." : "Add a concise headline and CTA.",
    (ml?.brands?.length ? "Feature brand mark more prominently." : "Include subtle brand cues early."),
    "Test color contrast for legibility and attention."
  ];

  return JSON.stringify({
    emotion,
    insight_summary:
      media === "video"
        ? `Video conveys a ${emotion} tone with ${ml?.video_emotions?.summary?.avg_faces_per_sec ?? 0} faces/sec on average.`
        : `Image conveys a ${emotion} tone with layout balance ${(ml?.layout_balance ?? 0).toFixed?.(2) ?? ml?.layout_balance}.`,
    weakness,
    suggestions
  });
}

/** Strip markdown fences and extract the first complete {...} JSON block */
function extractJsonBlock(text) {
  if (!text) return null;
  // Remove markdown fences
  let s = text.trim();
  if (s.startsWith("```")) {
    s = s.replace(/^```[a-zA-Z]*\s*/i, "").replace(/```[\s]*$/i, "").trim();
  }
  // Find the outermost { ... } in case the model adds prose
  const start = s.indexOf("{");
  const end   = s.lastIndexOf("}");
  if (start !== -1 && end > start) s = s.slice(start, end + 1);
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}

/** Main Gemini insight builder with fallbacks */
async function buildGeminiInsight(mlJson) {
  if (!GEMINI_API_KEY) return "Insight not available: GEMINI_API_KEY not configured";

  const ai = makeGemini();
  const context = compactContext(mlJson);

  const basePrompt = `
You are an expert advertising creative analyst. Analyse the ad metrics below and respond with ONLY a JSON object — no markdown, no prose outside the JSON.

Required JSON schema (fill every field, keep values concise):
{
  "emotion": "<dominant emotional tone of the ad>",
  "insight_summary": "<2-3 sentence strategic summary for a media buyer>",
  "weakness": "<one specific improvement area>",
  "suggestions": ["<actionable suggestion 1>", "<actionable suggestion 2>", "<actionable suggestion 3>"]
}

Ad metrics:
${JSON.stringify(context, null, 2)}
  `.trim();

  // 1024 tokens + thinking disabled so budget isn't consumed before output
  const generationConfig = {
    temperature: 0.3,
    topP: 0.95,
    maxOutputTokens: 1024,
    thinkingConfig: { thinkingBudget: 0 },
  };

  let lastErr = null;
  for (const modelName of MODEL_CANDIDATES) {
    try {
      const resp = await ai.models.generateContent({
        model: modelName,
        contents: [{ role: "user", parts: [{ text: basePrompt }] }],
        config: generationConfig,
      });
      const text = extractGenAIText(resp);

      if (text && text.trim()) {
        const obj = extractJsonBlock(text);
        if (obj && obj.insight_summary) {
          console.log(`✅ Gemini insight OK (${modelName})`);
          return JSON.stringify(obj);
        }
        // Model returned text but not parseable JSON → wrap as summary
        console.warn(`⚠️  Gemini (${modelName}) non-JSON response, wrapping.`);
        return JSON.stringify({ insight_summary: text.slice(0, 500) });
      }

      lastErr = new Error("Empty model output");
    } catch (e) {
      console.warn(`⚠️  Gemini model ${modelName} failed: ${e?.message}`);
      lastErr = e;
    }
  }

  // Ultimate fallback: local heuristic (always returns something)
  console.warn("Gemini unavailable → using heuristic fallback");
  return fallbackHeuristicInsight(mlJson);
}

/* -----------------------------------------------------------
 * Upload + analyze
 * --------------------------------------------------------- */
router.post("/upload", (req, res, next) => {
  upload.single("file")(req, res, function onMulter(err) {
    if (err?.code === "LIMIT_FILE_SIZE") {
      return res.status(413).json({ error: `File too large. Limit is ${MAX_UPLOAD_MB} MB.` });
    }
    if (err) return next(err);
    if (!req.file) return res.status(400).json({ error: "No file uploaded" });

    (async () => {
      const { mimetype, path: filePath, filename, originalname = filename, size } = req.file;
      const force = (req.query.force || "").toString().toLowerCase();
      const isVideo = looksLikeVideo({ mimetype, filename: originalname });
      const isImage = looksLikeImage({ mimetype, filename: originalname });
      const mlUrl = pickMlEndpoint({ isVideo, forceType: ["image", "video"].includes(force) ? force : undefined });

      console.log("🟢 /api/upload");
      console.log("   ↳ file:", originalname, "| type:", mimetype, "| size:", size, "bytes");
      console.log("   ↳ ML endpoint:", mlUrl);

      if (!isVideo && !isImage && !force) {
        try { fs.unlinkSync(filePath); } catch {}
        return res.status(400).json({ error: "Unsupported file type", detail: { mimetype, filename: originalname } });
      }

      try {
        const form = new FormData();
        form.append("file", fs.createReadStream(filePath), { filename: originalname, contentType: mimetype });

        const params = {};
        if (req.query.fps) params.fps = String(req.query.fps);

        const { data, status } = await axios.post(mlUrl, form, {
          headers: form.getHeaders(),
          maxBodyLength: Infinity,
          maxContentLength: Infinity,
          timeout: 1000 * 60 * 10,
          params,
          validateStatus: () => true,
        });

        try { fs.unlinkSync(filePath); } catch {}

        if (status < 200 || status >= 300) {
          console.error("🔥 ML non-2xx:", status, data);
          return res.status(500).json({ error: "Analysis failed", detail: data });
        }
        if (typeof data === "object" && data?.error) {
          console.error("🔥 ML error body:", data);
          return res.status(500).json({ error: "Analysis failed", detail: data });
        }

        // Rewrite heatmap filenames to full URLs (ML engine serves its own outputs)
        let mlData = { ...data };
        if (mlData.heatmap_url && !String(mlData.heatmap_url).startsWith("http")) {
          mlData.heatmap_url = `${ML_BASE}/outputs/${mlData.heatmap_url}`;
        }
        if (Array.isArray(mlData.keyframe_heatmaps)) {
          mlData.keyframe_heatmaps = mlData.keyframe_heatmaps.map((h) =>
            String(h).startsWith("http") ? h : `${ML_BASE}/outputs/${h}`
          );
        }

        // Always attach an insight (Gemini → fallback heuristic)
        let enriched = mlData;
        try {
          const insight = await buildGeminiInsight(mlData);
          enriched = { ...mlData, insight };
        } catch (e) {
          enriched = { ...mlData, insight: fallbackHeuristicInsight(mlData) };
        }

        return res.json(enriched);
      } catch (err) {
        try { fs.unlinkSync(filePath); } catch {}
        const status = err?.response?.status || 500;
        const body = err?.response?.data;
        console.error("🔥 /api/upload error:", { status, message: err?.message, data: body });
        return res.status(500).json({ error: "Analysis failed", detail: body || err?.message || String(err) });
      }
    })().catch(next);
  });
});

/* -----------------------------------------------------------
 * Health
 * --------------------------------------------------------- */
router.get("/limits", (_req, res) => {
  res.json({
    max_upload_mb: MAX_UPLOAD_MB,
    ml_base: ML_BASE,
    image_exts: [...IMAGE_EXTS],
    video_exts: [...VIDEO_EXTS],
    gemini_enabled: Boolean(GEMINI_API_KEY),
    gemini_model: PREFERRED_MODEL,
  });
});

router.get("/gemini/health", async (_req, res) => {
  if (!GEMINI_API_KEY) return res.json({ ok: false, reason: "no_api_key" });
  try {
    const ai = makeGemini();
    const r = await ai.models.generateContent({
      model: MODEL_CANDIDATES[0],
      contents: "ping",
      config: { maxOutputTokens: 5, temperature: 0 },
    });
    const text = extractGenAIText(r) || "ok";
    res.json({ ok: true, model: MODEL_CANDIDATES[0], reply: text });
  } catch (e) {
    res.json({ ok: false, model: MODEL_CANDIDATES[0], error: e?.message || String(e) });
  }
});

export default router;
