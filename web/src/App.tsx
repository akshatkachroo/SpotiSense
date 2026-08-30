import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowRight,
  BarChart3,
  BookHeart,
  BrainCircuit,
  Check,
  ChevronRight,
  CircleHelp,
  Clock3,
  Database,
  ExternalLink,
  Heart,
  History,
  Library,
  ListMusic,
  LoaderCircle,
  Menu,
  Music2,
  Plus,
  Radio,
  Server,
  ShieldCheck,
  Sparkles,
  UserRound,
  X,
  Zap,
} from "lucide-react";
import { APIError, api } from "./lib/api";
import { analyzeEmotion } from "./lib/emotion";
import { emotions, type Emotion, type EmotionPrediction, type Playlist, type Profile, type Recommendation, type Track } from "./types";

type View = "discover" | "playlists" | "history" | "about";
type Toast = { id: number; message: string; tone: "success" | "error" };

const sampleMoods = [
  "I finally finished a difficult week and feel proud, relieved, and ready to celebrate.",
  "The rain is soft, my mind is quiet, and I want a peaceful evening.",
  "I miss someone important and today feels heavier than usual.",
];

const emotionMeta: Record<Emotion, { label: string; color: string; description: string }> = {
  angry: { label: "Angry", color: "#ff715b", description: "intense and forceful" },
  calm: { label: "Calm", color: "#73d2de", description: "soft and grounded" },
  fear: { label: "Fear", color: "#a78bfa", description: "tense and atmospheric" },
  happy: { label: "Happy", color: "#f8d66d", description: "bright and energetic" },
  love: { label: "Love", color: "#ff8fab", description: "warm and connected" },
  sad: { label: "Sad", color: "#74a7ff", description: "reflective and low-key" },
};

function readStoredProfile(): Profile | null {
  try {
    const raw = localStorage.getItem("spotisense.profile");
    return raw ? (JSON.parse(raw) as Profile) : null;
  } catch {
    return null;
  }
}

function App() {
  const [profile, setProfile] = useState<Profile | null>(readStoredProfile);
  const [view, setView] = useState<View>("discover");
  const [mobileNav, setMobileNav] = useState(false);
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");
  const [playlists, setPlaylists] = useState<Playlist[]>([]);
  const [history, setHistory] = useState<Recommendation[]>([]);
  const [toasts, setToasts] = useState<Toast[]>([]);

  const notify = useCallback((message: string, tone: Toast["tone"] = "success") => {
    const id = Date.now();
    setToasts((current) => [...current, { id, message, tone }]);
    window.setTimeout(() => setToasts((current) => current.filter((toast) => toast.id !== id)), 3600);
  }, []);

  const refreshPlaylists = useCallback(async () => {
    if (!profile) return;
    try {
      setPlaylists(await api.playlists(profile.id));
    } catch (error) {
      notify(errorMessage(error), "error");
    }
  }, [notify, profile]);

  const refreshHistory = useCallback(async () => {
    if (!profile) return;
    try {
      setHistory(await api.history(profile.id));
    } catch (error) {
      notify(errorMessage(error), "error");
    }
  }, [notify, profile]);

  useEffect(() => {
    api.health().then(() => setApiStatus("online")).catch(() => setApiStatus("offline"));
  }, []);

  useEffect(() => {
    if (!profile) return;
    api.profile(profile.id).catch((error) => {
      if (error instanceof APIError && error.status === 404) {
        localStorage.removeItem("spotisense.profile");
        setProfile(null);
      }
    });
    void refreshPlaylists();
  }, [profile, refreshPlaylists]);

  useEffect(() => {
    if (view === "history") void refreshHistory();
    if (view === "playlists") void refreshPlaylists();
  }, [refreshHistory, refreshPlaylists, view]);

  const saveProfile = (nextProfile: Profile) => {
    localStorage.setItem("spotisense.profile", JSON.stringify(nextProfile));
    setProfile(nextProfile);
    setApiStatus("online");
  };

  const navigate = (nextView: View) => {
    setView(nextView);
    setMobileNav(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  if (!profile) {
    return (
      <>
        <Onboarding onCreated={saveProfile} apiStatus={apiStatus} />
        <ToastRegion toasts={toasts} />
      </>
    );
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileNav ? "sidebar-open" : ""}`}>
        <div className="brand">
          <span className="brand-mark"><Radio size={20} /></span>
          <span>SpotiSense</span>
        </div>
        <button className="mobile-close icon-button" onClick={() => setMobileNav(false)} aria-label="Close menu"><X /></button>
        <nav aria-label="Main navigation">
          <NavButton active={view === "discover"} icon={<Sparkles />} label="Discover" onClick={() => navigate("discover")} />
          <NavButton active={view === "playlists"} icon={<Library />} label="Your playlists" count={playlists.length} onClick={() => navigate("playlists")} />
          <NavButton active={view === "history"} icon={<History />} label="Listening history" onClick={() => navigate("history")} />
          <NavButton active={view === "about"} icon={<CircleHelp />} label="How it works" onClick={() => navigate("about")} />
        </nav>
        <div className="sidebar-bottom">
          <div className="service-status">
            <span className={`status-dot ${apiStatus}`} />
            <div><strong>{apiStatus === "online" ? "API online" : apiStatus === "checking" ? "Checking API" : "API waking"}</strong><span>Go recommendation service</span></div>
          </div>
          <div className="profile-chip"><span><UserRound size={18} /></span><div><strong>{profile.display_name}</strong><small>Demo listener</small></div></div>
        </div>
      </aside>
      {mobileNav && <button className="nav-scrim" onClick={() => setMobileNav(false)} aria-label="Close navigation" />}
      <main className="main-content">
        <header className="mobile-header">
          <div className="brand"><span className="brand-mark"><Radio size={18} /></span><span>SpotiSense</span></div>
          <button className="icon-button" onClick={() => setMobileNav(true)} aria-label="Open menu"><Menu /></button>
        </header>
        {view === "discover" && <Discover profile={profile} playlists={playlists} onPlaylistsChanged={refreshPlaylists} onRecommended={refreshHistory} notify={notify} />}
        {view === "playlists" && <PlaylistsView profile={profile} playlists={playlists} refresh={refreshPlaylists} notify={notify} />}
        {view === "history" && <HistoryView history={history} />}
        {view === "about" && <AboutView />}
      </main>
      <ToastRegion toasts={toasts} />
    </div>
  );
}

function Onboarding({ onCreated, apiStatus }: { onCreated: (profile: Profile) => void; apiStatus: "checking" | "online" | "offline" }) {
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!name.trim()) return setError("Enter a name to continue.");
    setLoading(true);
    setError("");
    try {
      onCreated(await api.createProfile(name.trim()));
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="landing">
      <nav className="landing-nav">
        <div className="brand"><span className="brand-mark"><Radio size={20} /></span><span>SpotiSense</span></div>
        <div className="privacy-pill"><ShieldCheck size={16} /> Mood text stays private</div>
      </nav>
      <div className="landing-glow glow-one" /><div className="landing-glow glow-two" />
      <section className="landing-grid">
        <div className="landing-copy">
          <div className="eyebrow"><span /> Emotion-aware music discovery</div>
          <h1>Hear what<br />you <em>feel.</em></h1>
          <p>Describe your headspace. SpotiSense reads the emotion and builds an explainable soundtrack from a locally ranked music catalog.</p>
          <div className="feature-row">
            <span><BrainCircuit /> Browser ML</span><span><Server /> Go API</span><span><Database /> PostgreSQL + Redis</span>
          </div>
        </div>
        <form className="onboarding-card" onSubmit={submit}>
          <div className="card-icon"><Music2 /></div>
          <p className="card-kicker">Your listening space</p>
          <h2>A playlist starts with a sentence.</h2>
          <p>Create a lightweight profile to keep recommendations, history, and playlists together.</p>
          <label htmlFor="display-name">What should we call you?</label>
          <input id="display-name" value={name} onChange={(event) => setName(event.target.value)} maxLength={80} placeholder="e.g. Akshat" autoFocus />
          {error && <div className="form-error" role="alert">{error}</div>}
          <button className="primary-button" type="submit" disabled={loading}>
            {loading ? <><LoaderCircle className="spin" /> {apiStatus === "offline" ? "Waking the API…" : "Starting…"}</> : <>Start listening <ArrowRight /></>}
          </button>
          <small><ShieldCheck size={14} /> Your written mood is analyzed in your browser and never stored.</small>
        </form>
      </section>
    </main>
  );
}

function Discover({ profile, playlists, onPlaylistsChanged, onRecommended, notify }: {
  profile: Profile;
  playlists: Playlist[];
  onPlaylistsChanged: () => Promise<void>;
  onRecommended: () => Promise<void>;
  notify: (message: string, tone?: Toast["tone"]) => void;
}) {
  const [text, setText] = useState("");
  const [limit, setLimit] = useState(6);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState("Reading your mood");
  const [modelProgress, setModelProgress] = useState<number>();
  const [prediction, setPrediction] = useState<EmotionPrediction>();
  const [recommendation, setRecommendation] = useState<Recommendation>();

  const discover = async () => {
    if (!text.trim()) return notify("Describe how you feel first.", "error");
    setLoading(true);
    setPrediction(undefined);
    setRecommendation(undefined);
    setProgress("Reading your mood");
    setModelProgress(undefined);
    try {
      const nextPrediction = await analyzeEmotion(text, (message, value) => {
        setProgress(message);
        setModelProgress(value);
      });
      setPrediction(nextPrediction);
      setProgress("Matching the music catalog");
      const nextRecommendation = await api.recommend({
        profileId: profile.id,
        emotion: nextPrediction.emotion,
        confidence: nextPrediction.confidence,
        context: text,
        limit,
      });
      setRecommendation(nextRecommendation);
      void onRecommended();
    } catch (error) {
      notify(errorMessage(error), "error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page discover-page">
      <section className="page-heading">
        <div><div className="eyebrow"><span /> Discover</div><h1>Good {timeGreeting()}, {profile.display_name}.</h1><p>Tell us where your head is at and we’ll find the sound to meet you there.</p></div>
        <div className="privacy-pill"><ShieldCheck size={16} /> Private by design</div>
      </section>
      <section className="mood-composer">
        <div className="composer-top"><label htmlFor="mood-text">How are you feeling right now?</label><span>{text.length} / 2,000</span></div>
        <textarea id="mood-text" value={text} onChange={(event) => setText(event.target.value)} maxLength={2000} placeholder="I finally finished a hard week and feel proud, relieved, and ready to celebrate…" />
        <div className="sample-row"><span>Not sure where to start?</span>{sampleMoods.map((sample, index) => <button key={sample} onClick={() => setText(sample)}>Try example {index + 1}</button>)}</div>
        <div className="composer-actions">
          <label className="track-count">Tracks <input type="range" min="3" max="12" value={limit} onChange={(event) => setLimit(Number(event.target.value))} /><strong>{limit}</strong></label>
          <button className="primary-button" onClick={discover} disabled={loading || !text.trim()}>{loading ? <><LoaderCircle className="spin" /> {progress}</> : <><Sparkles /> Find my soundtrack</>}</button>
        </div>
        {loading && modelProgress !== undefined && <div className="model-progress"><span style={{ width: `${Math.min(modelProgress, 100)}%` }} /></div>}
      </section>
      {!recommendation || !prediction ? <EmptyRecommendations /> : (
        <section className="results-section">
          <EmotionSummary prediction={prediction} cached={recommendation.cached} />
          <div className="section-heading"><div><span className="section-kicker">Made for this moment</span><h2>Your mood mix</h2></div><span>{recommendation.tracks.length} tracks · artist-diverse ranking</span></div>
          <div className="track-grid">
            {recommendation.tracks.map((track, index) => <TrackCard key={track.id} track={track} index={index} playlists={playlists} onSaved={async (playlistId) => { try { await api.addTrack(playlistId, track.id); await onPlaylistsChanged(); notify(`Saved “${track.name}” to your playlist.`); } catch (error) { notify(errorMessage(error), "error"); } }} />)}
          </div>
        </section>
      )}
    </div>
  );
}

function EmotionSummary({ prediction, cached }: { prediction: EmotionPrediction; cached: boolean }) {
  const sorted = useMemo(() => [...emotions].sort((a, b) => prediction.scores[b] - prediction.scores[a]), [prediction]);
  const meta = emotionMeta[prediction.emotion];
  return (
    <div className="emotion-summary" style={{ "--emotion-color": meta.color } as React.CSSProperties}>
      <div className="emotion-primary"><div className="emotion-orbit"><span><Activity /></span></div><div><span className="section-kicker">Mood detected</span><h2>{meta.label}</h2><p>{Math.round(prediction.confidence * 100)}% confidence · {meta.description}</p></div></div>
      <div className="score-list">{sorted.map((emotion) => <div className="score-row" key={emotion}><span>{emotionMeta[emotion].label}</span><div><i style={{ width: `${prediction.scores[emotion] * 100}%`, background: emotionMeta[emotion].color }} /></div><strong>{Math.round(prediction.scores[emotion] * 100)}%</strong></div>)}</div>
      <div className="inference-meta"><span><BrainCircuit size={16} /> {prediction.backend === "browser-transformer" ? "Fine-tuned browser model" : "Development fallback"}</span><span><Zap size={16} /> {cached ? "Redis cache hit" : "Fresh ranking"}</span>{prediction.fallbackReason && <span title={prediction.fallbackReason}><CircleHelp size={16} /> Browser model unavailable</span>}</div>
    </div>
  );
}

function EmptyRecommendations() {
  return <section className="empty-results"><div className="vinyl"><span /></div><div><span className="section-kicker">Waiting for your signal</span><h2>Your recommendations will appear here.</h2><p>You’ll see the complete emotion breakdown, match explanations, and direct Spotify search links.</p></div></section>;
}

function TrackCard({ track, index, playlists, onSaved }: { track: Track; index: number; playlists: Playlist[]; onSaved: (playlistId: string) => Promise<void> }) {
  const [playlistId, setPlaylistId] = useState(playlists[0]?.id ?? "");
  const [saving, setSaving] = useState(false);
  useEffect(() => { if (!playlistId && playlists[0]) setPlaylistId(playlists[0].id); }, [playlistId, playlists]);
  const features = [
    ["Valence", track.valence], ["Energy", track.energy], ["Dance", track.danceability],
    ["Acoustic", track.acousticness], ["Instrumental", track.instrumentalness],
  ] as const;
  const color = emotionMeta[track.primary_emotion].color;
  return (
    <article className="track-card" style={{ "--track-color": color } as React.CSSProperties}>
      <div className="album-art">{track.image_url ? <img src={track.image_url} alt="" /> : <><Music2 /><span>{String(index + 1).padStart(2, "0")}</span></>}</div>
      <div className="track-body"><div className="match-pill"><i /> {Math.round(track.match_score * 100)}% mood match</div><h3>{track.name}</h3><p>{track.artist}</p>
        <details><summary><BarChart3 size={15} /> Why this track <ChevronRight size={15} /></summary><div className="feature-bars">{features.map(([name, value]) => <div key={name}><span>{name}</span><i><b style={{ width: `${value * 100}%` }} /></i><strong>{Math.round(value * 100)}</strong></div>)}</div><small>Lyric-derived mood proxies, not Spotify Audio Features.</small></details>
        <div className="track-actions"><a href={track.spotify_url} target="_blank" rel="noreferrer">Open Spotify <ExternalLink /></a><div className="save-control"><select aria-label={`Playlist for ${track.name}`} value={playlistId} onChange={(event) => setPlaylistId(event.target.value)} disabled={!playlists.length}>{playlists.length ? playlists.map((playlist) => <option key={playlist.id} value={playlist.id}>{playlist.name}</option>) : <option>Create a playlist first</option>}</select><button aria-label={`Save ${track.name}`} disabled={!playlistId || saving} onClick={async () => { setSaving(true); try { await onSaved(playlistId); } finally { setSaving(false); } }}>{saving ? <LoaderCircle className="spin" /> : <Heart />}</button></div></div>
      </div>
    </article>
  );
}

function PlaylistsView({ profile, playlists, refresh, notify }: { profile: Profile; playlists: Playlist[]; refresh: () => Promise<void>; notify: (message: string, tone?: Toast["tone"]) => void }) {
  const [name, setName] = useState("");
  const [creating, setCreating] = useState(false);
  const submit = async (event: FormEvent) => {
    event.preventDefault(); if (!name.trim()) return;
    setCreating(true);
    try { await api.createPlaylist(profile.id, name.trim()); setName(""); await refresh(); notify("Playlist created."); }
    catch (error) { notify(errorMessage(error), "error"); }
    finally { setCreating(false); }
  };
  return <div className="page"><section className="page-heading"><div><div className="eyebrow"><span /> Your library</div><h1>Playlists</h1><p>Keep the recommendations that deserve another listen.</p></div><form className="inline-form" onSubmit={submit}><input value={name} onChange={(event) => setName(event.target.value)} maxLength={80} placeholder="Late-night reset" aria-label="New playlist name" /><button className="primary-button" disabled={creating || !name.trim()}><Plus /> Create</button></form></section>
    {!playlists.length ? <div className="empty-library"><span><ListMusic /></span><h2>No playlists yet</h2><p>Create one here, then save tracks while exploring recommendations.</p></div> : <div className="playlist-grid">{playlists.map((playlist) => <article className="playlist-card" key={playlist.id}><div className="playlist-cover"><ListMusic /><span>{playlist.tracks.length}</span></div><div><h2>{playlist.name}</h2><p>{playlist.tracks.length} {playlist.tracks.length === 1 ? "track" : "tracks"}</p>{playlist.tracks.length ? <ol>{playlist.tracks.slice(0, 5).map((track) => <li key={track.id}><span>{track.name}</span><small>{track.artist}</small><a href={track.spotify_url} target="_blank" rel="noreferrer" aria-label={`Open ${track.name} on Spotify`}><ExternalLink /></a></li>)}</ol> : <div className="playlist-empty">Save a recommendation to start this playlist.</div>}</div></article>)}</div>}
  </div>;
}

function HistoryView({ history }: { history: Recommendation[] }) {
  return <div className="page"><section className="page-heading"><div><div className="eyebrow"><span /> Past sessions</div><h1>Listening history</h1><p>Detected moods and results are retained. Your written descriptions are not.</p></div><div className="privacy-pill"><ShieldCheck size={16} /> Context never stored</div></section>
    {!history.length ? <div className="empty-library"><span><Clock3 /></span><h2>No sessions yet</h2><p>Your recommendation history will appear after your first mood mix.</p></div> : <div className="timeline">{history.map((session) => <article key={session.id}><div className="timeline-marker" style={{ background: emotionMeta[session.emotion].color }} /><div className="history-meta"><span>{formatDate(session.created_at)}</span><strong style={{ color: emotionMeta[session.emotion].color }}>{emotionMeta[session.emotion].label}</strong><small>{Math.round(session.confidence * 100)}% confidence</small></div><div className="history-tracks">{session.tracks.map((track) => <a key={track.id} href={track.spotify_url} target="_blank" rel="noreferrer"><span><Music2 /></span><div><strong>{track.name}</strong><small>{track.artist}</small></div><ExternalLink /></a>)}</div></article>)}</div>}
  </div>;
}

function AboutView() {
  const steps = [
    { icon: <BrainCircuit />, title: "Understand the feeling", copy: "A compact transformer runs in a Web Worker and returns six emotion probabilities without sending the written mood to a server." },
    { icon: <Server />, title: "Rank the catalog", copy: "The Go API maps the dominant emotion to target mood characteristics and computes a weighted match over the offline catalog." },
    { icon: <BookHeart />, title: "Build a listening memory", copy: "PostgreSQL stores profiles, history, and playlists while Redis caches repeat feeds and optional Spotify metadata lookups." },
  ];
  return <div className="page about-page"><section className="page-heading"><div><div className="eyebrow"><span /> Under the hood</div><h1>Explainable by design.</h1><p>Every recommendation follows a small, inspectable path—no listening-history access required.</p></div></section><div className="architecture-flow">{steps.map((step, index) => <div className="architecture-step" key={step.title}><span>{step.icon}</span><small>0{index + 1}</small><h2>{step.title}</h2><p>{step.copy}</p>{index < steps.length - 1 && <ChevronRight className="flow-arrow" />}</div>)}</div><section className="principles"><div><ShieldCheck /><h3>Privacy</h3><p>The browser analyzes raw mood text; the API receives only the result and a SHA-256 context key.</p></div><div><Activity /><h3>Resilience</h3><p>Local catalog ranking and graceful fallbacks keep discovery useful without Spotify or managed infrastructure.</p></div><div><BarChart3 /><h3>Transparency</h3><p>Emotion probabilities, match scores, data provenance, and cache status remain visible to the listener.</p></div></section></div>;
}

function NavButton({ active, icon, label, count, onClick }: { active: boolean; icon: React.ReactNode; label: string; count?: number; onClick: () => void }) {
  return <button className={active ? "active" : ""} onClick={onClick}>{icon}<span>{label}</span>{count !== undefined && <small>{count}</small>}</button>;
}

function ToastRegion({ toasts }: { toasts: Toast[] }) {
  return <div className="toast-region" aria-live="polite">{toasts.map((toast) => <div className={`toast ${toast.tone}`} key={toast.id}>{toast.tone === "success" ? <Check /> : <X />} {toast.message}</div>)}</div>;
}

function errorMessage(error: unknown): string {
  if (error instanceof TypeError) return "The API is unavailable. If it is on a free tier, give it a moment to wake up.";
  return error instanceof Error ? error.message : "Something unexpected happened.";
}

function timeGreeting(): string {
  const hour = new Date().getHours();
  if (hour < 12) return "morning";
  if (hour < 18) return "afternoon";
  return "evening";
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit" }).format(new Date(value));
}

export default App;
