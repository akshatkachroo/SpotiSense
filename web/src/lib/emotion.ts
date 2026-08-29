import { emotions, type Emotion, type EmotionPrediction } from "../types";

type PendingRequest = {
  resolve: (prediction: EmotionPrediction) => void;
  reject: (error: Error) => void;
  onProgress?: (message: string, progress?: number) => void;
  timeout: number;
};

type WorkerResponse =
  | { id: string; type: "progress"; message: string; progress?: number }
  | { id: string; type: "result"; scores: Partial<Record<Emotion, number>> }
  | { id: string; type: "error"; error: string };

const modelPath = import.meta.env.VITE_EMOTION_MODEL ?? "/models/spotisense-emotion";
const pending = new Map<string, PendingRequest>();
let worker: Worker | undefined;

const vocabulary: Record<Emotion, string[]> = {
  angry: [
    "angry", "annoyed", "betrayed", "furious", "frustrated", "hate", "irritated", "mad",
    "outraged", "resentful", "unfair",
  ],
  calm: [
    "balanced", "calm", "centered", "content", "grounded", "peaceful", "quiet", "relaxed",
    "relieved", "safe", "serene", "steady",
  ],
  fear: [
    "afraid", "anxious", "danger", "dread", "fear", "frightened", "nervous", "panic",
    "scared", "terrified", "threatened", "uneasy", "worried",
  ],
  happy: [
    "celebrate", "cheerful", "delighted", "excited", "glad", "great", "happy", "joy",
    "optimistic", "proud", "smiling", "wonderful",
  ],
  love: [
    "adore", "affection", "care", "cherish", "connected", "family", "friendship", "grateful",
    "heart", "kindness", "love", "partner", "tender",
  ],
  sad: [
    "alone", "crying", "dejected", "disappointed", "empty", "grief", "heartbroken", "hopeless",
    "lonely", "loss", "miss", "sad", "tired",
  ],
};

function getWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL("../workers/emotion.worker.ts", import.meta.url), { type: "module" });
  worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
    const request = pending.get(data.id);
    if (!request) return;
    if (data.type === "progress") {
      request.onProgress?.(data.message, data.progress);
      return;
    }
    window.clearTimeout(request.timeout);
    pending.delete(data.id);
    if (data.type === "error") {
      request.reject(new Error(data.error));
      return;
    }
    request.resolve(toPrediction(data.scores, "browser-transformer"));
  };
  worker.onerror = (event) => {
    const error = new Error(event.message || "The emotion worker stopped unexpectedly.");
    for (const request of pending.values()) {
      window.clearTimeout(request.timeout);
      request.reject(error);
    }
    pending.clear();
    worker?.terminate();
    worker = undefined;
  };
  return worker;
}

function toPrediction(
  rawScores: Partial<Record<Emotion, number>>,
  backend: EmotionPrediction["backend"],
  fallbackReason?: string,
): EmotionPrediction {
  const sum = emotions.reduce((total, emotion) => total + Math.max(rawScores[emotion] ?? 0, 0), 0);
  const denominator = sum || emotions.length;
  const scores = Object.fromEntries(
    emotions.map((emotion) => [emotion, sum ? Math.max(rawScores[emotion] ?? 0, 0) / denominator : 1 / denominator]),
  ) as Record<Emotion, number>;
  const emotion = emotions.reduce(
    (best, candidate) => (scores[candidate] > scores[best] ? candidate : best),
    "calm",
  );
  return { emotion, confidence: scores[emotion], scores, backend, fallbackReason };
}

export function lexicalPrediction(text: string, fallbackReason?: string): EmotionPrediction {
  const words = text.toLowerCase().match(/[a-z']+/g) ?? [];
  const rawScores = Object.fromEntries(emotions.map((emotion) => [emotion, 1])) as Record<Emotion, number>;
  for (const word of words) {
    for (const emotion of emotions) {
      if (vocabulary[emotion].includes(word)) rawScores[emotion] += 3;
    }
  }
  return toPrediction(rawScores, "lexical-fallback", fallbackReason);
}

function transformerPrediction(
  text: string,
  onProgress?: (message: string, progress?: number) => void,
): Promise<EmotionPrediction> {
  const id = crypto.randomUUID();
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      pending.delete(id);
      reject(new Error("Model loading timed out."));
    }, 120_000);
    pending.set(id, { resolve, reject, onProgress, timeout });
    getWorker().postMessage({ id, type: "analyze", text, modelPath });
  });
}

export async function analyzeEmotion(
  text: string,
  onProgress?: (message: string, progress?: number) => void,
): Promise<EmotionPrediction> {
  const cleanText = text.trim().replace(/\s+/g, " ");
  if (!cleanText) throw new Error("Describe how you feel first.");
  if (cleanText.length > 2_000) throw new Error("Your description must be 2,000 characters or fewer.");
  try {
    return await transformerPrediction(cleanText, onProgress);
  } catch (error) {
    const message = error instanceof Error ? error.message : "The browser model is unavailable.";
    onProgress?.("Using the lightweight offline analyzer");
    return lexicalPrediction(cleanText, message);
  }
}
