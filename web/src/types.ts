export const emotions = ["angry", "calm", "fear", "happy", "love", "sad"] as const;

export type Emotion = (typeof emotions)[number];

export interface Profile {
  id: string;
  display_name: string;
  created_at: string;
}

export interface Track {
  id: string;
  name: string;
  artist: string;
  primary_emotion: Emotion;
  valence: number;
  energy: number;
  danceability: number;
  acousticness: number;
  instrumentalness: number;
  match_score: number;
  spotify_url: string;
  image_url?: string;
  data_source: string;
}

export interface Recommendation {
  id: string;
  profile_id: string;
  emotion: Emotion;
  confidence: number;
  tracks: Track[];
  cached: boolean;
  created_at: string;
}

export interface Playlist {
  id: string;
  profile_id: string;
  name: string;
  tracks: Track[];
  created_at: string;
  updated_at: string;
}

export interface EmotionPrediction {
  emotion: Emotion;
  confidence: number;
  scores: Record<Emotion, number>;
  backend: "browser-transformer" | "lexical-fallback";
  fallbackReason?: string;
}
