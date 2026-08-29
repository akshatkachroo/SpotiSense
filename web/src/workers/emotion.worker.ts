/// <reference lib="webworker" />

import { env, pipeline } from "@huggingface/transformers";
import { emotions, type Emotion } from "../types";

type AnalyzeRequest = { id: string; type: "analyze"; text: string; modelPath: string };
type ModelOutput = { label: string; score: number };
type Classifier = (text: string, options: { top_k: null }) => Promise<unknown>;

const workerScope: DedicatedWorkerGlobalScope = self as unknown as DedicatedWorkerGlobalScope;
let classifierPromise: Promise<Classifier> | undefined;
let activeModel = "";

function normalizeLabel(label: string): Emotion | undefined {
  const normalized = label.toLowerCase().replace(/^label_/, "");
  if (emotions.includes(normalized as Emotion)) return normalized as Emotion;
  const aliases: Record<string, Emotion> = {
    anger: "angry",
    disgust: "angry",
    neutral: "calm",
    joy: "happy",
    sadness: "sad",
  };
  return aliases[normalized];
}

function loadClassifier(modelPath: string, requestId: string): Promise<Classifier> {
  if (classifierPromise && activeModel === modelPath) return classifierPromise;
  activeModel = modelPath;
  const local = modelPath.startsWith("/") || modelPath.startsWith(".");
  env.allowLocalModels = local;
  env.allowRemoteModels = !local;
  const createTextClassifier = pipeline as unknown as (
    task: "text-classification",
    model: string,
    options: Record<string, unknown>,
  ) => Promise<Classifier>;
  classifierPromise = createTextClassifier("text-classification", modelPath, {
    dtype: "q8",
    progress_callback: (event: { status?: string; progress?: number; file?: string }) => {
      const percentage = typeof event.progress === "number" ? event.progress : undefined;
      const file = event.file ? ` ${event.file.split("/").at(-1)}` : "";
      workerScope.postMessage({
        id: requestId,
        type: "progress",
        message: event.status === "ready" ? "Emotion model ready" : `Loading emotion model${file}`,
        progress: percentage,
      });
    },
  });
  return classifierPromise;
}

workerScope.onmessage = async ({ data }: MessageEvent<AnalyzeRequest>) => {
  if (data.type !== "analyze") return;
  try {
    const classifier = await loadClassifier(data.modelPath, data.id);
    const raw = await classifier(data.text, { top_k: null }) as ModelOutput[] | ModelOutput[][];
    const outputs = (Array.isArray(raw[0]) ? raw[0] : raw) as ModelOutput[];
    const scores: Partial<Record<Emotion, number>> = {};
    for (const output of outputs) {
      const emotion = normalizeLabel(output.label);
      if (emotion) scores[emotion] = (scores[emotion] ?? 0) + output.score;
    }
    if (Object.keys(scores).length === 0) throw new Error("The model returned unknown emotion labels.");
    workerScope.postMessage({ id: data.id, type: "result", scores });
  } catch (error) {
    classifierPromise = undefined;
    workerScope.postMessage({
      id: data.id,
      type: "error",
      error: error instanceof Error ? error.message : "Emotion inference failed.",
    });
  }
};

export {};
