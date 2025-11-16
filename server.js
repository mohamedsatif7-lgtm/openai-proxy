// server.js
import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import multer from "multer";
import fs from "fs";
import OpenAI from "openai";

dotenv.config();

const app = express();
app.use(cors()); // allow all origins by default (can be tightened later)
app.use(express.json({ limit: "1mb" }));
app.use(express.urlencoded({ extended: true }));

const upload = multer();

// OpenAI client (v4 style)
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

/* -----------------------
   Health / root
   ----------------------- */
app.get("/", (req, res) => {
  res.json({ status: "ok", ts: Date.now() });
});

/* -----------------------
   Translate (text) 
   Expected body: { text, targetLanguage }
   Returns: { result: "...translated text..." }
   ----------------------- */
app.post("/translate", async (req, res) => {
  try {
    const { text, targetLanguage } = req.body;
    if (!text || !targetLanguage) {
      return res.status(400).json({ error: "Missing text or targetLanguage" });
    }

    // Ask OpenAI to translate concisely
    const prompt = `Translate the following text to ${targetLanguage} and return only the translation:\n\n${text}`;

    const resp = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: "You are a helpful translator." },
        { role: "user", content: prompt }
      ],
      max_tokens: 500,
      temperature: 0.1
    });

    const translated = resp?.choices?.[0]?.message?.content?.trim() || "";
    res.json({ result: translated });
  } catch (err) {
    console.error("translate error:", err?.message || err);
    res.status(500).json({ error: "Translate failed", details: String(err?.message || err) });
  }
});

/* -----------------------
   Chat partner
   Expected body: { message, language }
   Returns: { reply: "..." }
   ----------------------- */
app.post("/chat", async (req, res) => {
  try {
    const { message, language } = req.body;
    if (!message) return res.status(400).json({ error: "Missing message" });

    const langNote = language ? `Respond in ${language}.` : "";
    const resp = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: `You are a friendly language partner. ${langNote}` },
        { role: "user", content: message }
      ],
      max_tokens: 400,
      temperature: 0.7
    });

    const reply = resp?.choices?.[0]?.message?.content?.trim() || "";
    res.json({ reply });
  } catch (err) {
    console.error("chat error:", err?.message || err);
    res.status(500).json({ error: "Chat failed", details: String(err?.message || err) });
  }
});

/* -----------------------
   Voice-to-Text (upload audio)
   Accepts multipart/form-data, field name "audio"
   Returns: { text: "transcribed text" }
   ----------------------- */
app.post("/voice-to-text", upload.single("audio"), async (req, res) => {
  try {
    if (!req.file || !req.file.buffer) return res.status(400).json({ error: "No audio file uploaded" });

    // write temp file (OpenAI audio transcription expects a file-like stream)
    const tmpPath = `/tmp/record-${Date.now()}.webm`;
    fs.writeFileSync(tmpPath, req.file.buffer);

    // call OpenAI transcription (Audio API)
    const transcription = await openai.audio.transcriptions.create({
      file: fs.createReadStream(tmpPath),
      model: "gpt-4o-mini-transcribe"
    });

    // cleanup
    try { fs.unlinkSync(tmpPath); } catch(e){}

    res.json({ text: transcription?.text || "" });
  } catch (err) {
    console.error("voice-to-text error:", err?.message || err);
    res.status(500).json({ error: "Voice-to-text failed", details: String(err?.message || err) });
  }
});

/* -----------------------
   Text-to-Voice (TTS)
   Expected body: { text }
   Returns audio/mpeg stream
   ----------------------- */
app.post("/text-to-voice", async (req, res) => {
  try {
    const { text } = req.body;
    if (!text) return res.status(400).json({ error: "Missing text" });

    // Use OpenAI audio.speech.create - returns a binary audio stream
    const speech = await openai.audio.speech.create({
      model: "gpt-4o-mini-tts",
      voice: "alloy", // example voice identifier; adjust if your account expects other params
      input: text
    });

    // speech is a Response-like object: convert to buffer
    const arrayBuffer = await speech.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    res.setHeader("Content-Type", "audio/mpeg");
    res.setHeader("Content-Length", buffer.length);
    res.send(buffer);
  } catch (err) {
    console.error("text-to-voice error:", err?.message || err);
    res.status(500).json({ error: "TTS failed", details: String(err?.message || err) });
  }
});

/* -----------------------
   Allow preflight for all routes
----------------------- */
app.options("/*", (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  res.sendStatus(200);
});

/* -----------------------
   Start server (use process.env.PORT)
----------------------- */
const PORT = Number(process.env.PORT || process.env.PORT_NUMBER || 10000);
app.listen(PORT, () => {
  console.log(`Server started on port ${PORT}`);
});
