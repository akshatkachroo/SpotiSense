import type { Emotion, Playlist, Profile, Recommendation } from "../types";

const API_URL = (import.meta.env.VITE_API_URL ?? "http://localhost:8080").replace(/\/$/, "");

type RequestOptions = RequestInit & { retries?: number };

export class APIError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "APIError";
  }
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { retries, ...fetchOptions } = options;
  const retryLimit = retries ?? ((fetchOptions.method ?? "GET").toUpperCase() === "GET" ? 2 : 0);
  let lastError: unknown;

  for (let attempt = 0; attempt <= retryLimit; attempt += 1) {
    try {
      const response = await fetch(`${API_URL}${path}`, {
        ...fetchOptions,
        headers: {
          Accept: "application/json",
          ...(fetchOptions.body ? { "Content-Type": "application/json" } : {}),
          ...fetchOptions.headers,
        },
      });
      const payload = (await response.json().catch(() => ({}))) as {
        error?: string | { message?: string };
      };
      if (!response.ok) {
        if (response.status >= 500 && attempt < retryLimit) {
          await delay(700 * 2 ** attempt);
          continue;
        }
        const message = typeof payload.error === "string" ? payload.error : payload.error?.message;
        throw new APIError(message ?? `Request failed with status ${response.status}`, response.status);
      }
      return payload as T;
    } catch (error) {
      lastError = error;
      if (error instanceof APIError || attempt === retryLimit) {
        throw error;
      }
      await delay(700 * 2 ** attempt);
    }
  }
  throw lastError instanceof Error ? lastError : new APIError("The API is unavailable.");
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

export const api = {
  health: () => request<{ status: string }>("/healthz", { retries: 3 }),
  createProfile: (displayName: string) =>
    request<Profile>("/v1/profiles", {
      method: "POST",
      body: JSON.stringify({ display_name: displayName }),
    }),
  profile: (profileId: string) => request<Profile>(`/v1/profiles/${profileId}`),
  recommend: async (input: {
    profileId: string;
    emotion: Emotion;
    confidence: number;
    context: string;
    limit: number;
  }) => {
    const contextKey = await hashContext(input.context);
    return request<Recommendation>("/v1/recommendations", {
      method: "POST",
      body: JSON.stringify({
        profile_id: input.profileId,
        emotion: input.emotion,
        confidence: input.confidence,
        context_key: contextKey,
        limit: input.limit,
      }),
    });
  },
  history: async (profileId: string) => {
    const result = await request<{ history: Recommendation[] }>(
      `/v1/profiles/${profileId}/history?limit=30`,
    );
    return result.history;
  },
  playlists: async (profileId: string) => {
    const result = await request<{ playlists: Playlist[] }>(`/v1/profiles/${profileId}/playlists`);
    return result.playlists;
  },
  createPlaylist: (profileId: string, name: string) =>
    request<Playlist>("/v1/playlists", {
      method: "POST",
      body: JSON.stringify({ profile_id: profileId, name }),
    }),
  addTrack: (playlistId: string, trackId: string) =>
    request<Playlist>(`/v1/playlists/${playlistId}/tracks`, {
      method: "POST",
      body: JSON.stringify({ track_id: trackId }),
    }),
};

async function hashContext(context: string): Promise<string> {
  const encoded = new TextEncoder().encode(context.trim());
  const digest = await crypto.subtle.digest("SHA-256", encoded);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}
