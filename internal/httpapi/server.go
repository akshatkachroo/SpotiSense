package httpapi

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/akshatkachroo/spotisense/internal/cache"
	"github.com/akshatkachroo/spotisense/internal/catalog"
	"github.com/akshatkachroo/spotisense/internal/model"
	"github.com/akshatkachroo/spotisense/internal/spotify"
	"github.com/akshatkachroo/spotisense/internal/store"
)

type Server struct {
	store          store.Store
	cache          cache.Cache
	catalog        *catalog.Catalog
	spotify        *spotify.Client
	logger         *slog.Logger
	allowedOrigins map[string]struct{}
	startedAt      time.Time
	requests       atomic.Uint64
	cacheHits      atomic.Uint64
	cacheMisses    atomic.Uint64
}

func New(dataStore store.Store, responseCache cache.Cache, trackCatalog *catalog.Catalog, spotifyClient *spotify.Client, logger *slog.Logger, allowedOrigins ...string) *Server {
	origins := make(map[string]struct{}, len(allowedOrigins))
	for _, origin := range allowedOrigins {
		if clean := strings.TrimRight(strings.TrimSpace(origin), "/"); clean != "" {
			origins[clean] = struct{}{}
		}
	}
	return &Server{
		store:          dataStore,
		cache:          responseCache,
		catalog:        trackCatalog,
		spotify:        spotifyClient,
		logger:         logger,
		allowedOrigins: origins,
		startedAt:      time.Now().UTC(),
	}
}

func (server *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", server.health)
	mux.HandleFunc("GET /metrics", server.metrics)
	mux.HandleFunc("GET /v1/emotions", server.listEmotions)
	mux.HandleFunc("GET /v1/tracks", server.searchTracks)
	mux.HandleFunc("POST /v1/profiles", server.createProfile)
	mux.HandleFunc("GET /v1/profiles/{profileID}", server.getProfile)
	mux.HandleFunc("POST /v1/recommendations", server.recommend)
	mux.HandleFunc("GET /v1/profiles/{profileID}/history", server.listHistory)
	mux.HandleFunc("POST /v1/playlists", server.createPlaylist)
	mux.HandleFunc("GET /v1/profiles/{profileID}/playlists", server.listPlaylists)
	mux.HandleFunc("GET /v1/playlists/{playlistID}", server.getPlaylist)
	mux.HandleFunc("POST /v1/playlists/{playlistID}/tracks", server.addTrack)
	return server.middleware(mux)
}

func (server *Server) middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		started := time.Now()
		server.requests.Add(1)
		writer.Header().Set("X-Content-Type-Options", "nosniff")
		writer.Header().Set("Cache-Control", "no-store")
		server.applyCORS(writer, request)
		if request.Method == http.MethodOptions {
			writer.WriteHeader(http.StatusNoContent)
			return
		}
		writer.Header().Set("Content-Type", "application/json; charset=utf-8")
		request.Body = http.MaxBytesReader(writer, request.Body, 1<<20)
		next.ServeHTTP(writer, request)
		server.logger.Info("request", "method", request.Method, "path", request.URL.Path, "duration_ms", time.Since(started).Milliseconds())
	})
}

func (server *Server) applyCORS(writer http.ResponseWriter, request *http.Request) {
	origin := strings.TrimRight(strings.TrimSpace(request.Header.Get("Origin")), "/")
	if origin == "" {
		return
	}
	if _, wildcard := server.allowedOrigins["*"]; !wildcard {
		if _, allowed := server.allowedOrigins[origin]; !allowed {
			return
		}
	}
	writer.Header().Add("Vary", "Origin")
	writer.Header().Set("Access-Control-Allow-Origin", origin)
	writer.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	writer.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	writer.Header().Set("Access-Control-Max-Age", "600")
}

func (server *Server) health(writer http.ResponseWriter, _ *http.Request) {
	writeJSON(writer, http.StatusOK, map[string]any{
		"status":             "ok",
		"catalog_tracks":     server.catalog.Count(),
		"persistence":        server.store.Backend(),
		"cache":              server.cache.Backend(),
		"spotify_enrichment": server.spotify.Enabled(),
		"uptime_seconds":     int(time.Since(server.startedAt).Seconds()),
	})
}

func (server *Server) metrics(writer http.ResponseWriter, _ *http.Request) {
	writeJSON(writer, http.StatusOK, map[string]uint64{
		"requests_total":            server.requests.Load(),
		"recommendation_cache_hit":  server.cacheHits.Load(),
		"recommendation_cache_miss": server.cacheMisses.Load(),
	})
}

func (server *Server) listEmotions(writer http.ResponseWriter, _ *http.Request) {
	writeJSON(writer, http.StatusOK, map[string]any{"emotions": []map[string]string{
		{"id": "happy", "description": "bright, upbeat and energetic"},
		{"id": "sad", "description": "low-valence and reflective"},
		{"id": "angry", "description": "intense, forceful and high-energy"},
		{"id": "calm", "description": "soft, acoustic and low-energy"},
		{"id": "fear", "description": "tense, atmospheric and moderate-energy"},
		{"id": "love", "description": "warm, positive and balanced"},
	}})
}

func (server *Server) searchTracks(writer http.ResponseWriter, request *http.Request) {
	limit := queryLimit(request, 10)
	writeJSON(writer, http.StatusOK, map[string]any{"tracks": server.catalog.Search(request.URL.Query().Get("q"), limit)})
}

func (server *Server) createProfile(writer http.ResponseWriter, request *http.Request) {
	var input struct {
		DisplayName string `json:"display_name"`
	}
	if !decodeJSON(writer, request, &input) {
		return
	}
	input.DisplayName = strings.TrimSpace(input.DisplayName)
	if len(input.DisplayName) < 1 || len(input.DisplayName) > 80 {
		writeError(writer, http.StatusUnprocessableEntity, "display_name must contain 1 to 80 characters")
		return
	}
	profile, err := server.store.CreateProfile(request.Context(), input.DisplayName)
	if err != nil {
		server.internalError(writer, "create profile", err)
		return
	}
	writeJSON(writer, http.StatusCreated, profile)
}

func (server *Server) getProfile(writer http.ResponseWriter, request *http.Request) {
	profile, err := server.store.GetProfile(request.Context(), request.PathValue("profileID"))
	if errors.Is(err, store.ErrNotFound) {
		writeError(writer, http.StatusNotFound, "profile not found")
		return
	}
	if err != nil {
		server.internalError(writer, "get profile", err)
		return
	}
	writeJSON(writer, http.StatusOK, profile)
}

func (server *Server) recommend(writer http.ResponseWriter, request *http.Request) {
	var input struct {
		ProfileID  string  `json:"profile_id"`
		Emotion    string  `json:"emotion"`
		Confidence float64 `json:"confidence"`
		ContextKey string  `json:"context_key"`
		Limit      int     `json:"limit"`
	}
	if !decodeJSON(writer, request, &input) {
		return
	}
	if _, err := server.store.GetProfile(request.Context(), input.ProfileID); errors.Is(err, store.ErrNotFound) {
		writeError(writer, http.StatusNotFound, "profile not found")
		return
	} else if err != nil {
		server.internalError(writer, "validate profile", err)
		return
	}
	if input.Confidence < 0 || input.Confidence > 1 {
		writeError(writer, http.StatusUnprocessableEntity, "confidence must be between 0 and 1")
		return
	}
	input.ContextKey = strings.TrimSpace(input.ContextKey)
	if len(input.ContextKey) < 16 || len(input.ContextKey) > 128 {
		writeError(writer, http.StatusUnprocessableEntity, "context_key must contain 16 to 128 characters")
		return
	}
	if input.Limit < 1 || input.Limit > 20 {
		input.Limit = 5
	}
	input.Emotion = strings.ToLower(strings.TrimSpace(input.Emotion))

	cacheKey := recommendationCacheKey(input.Emotion, input.ContextKey, input.Limit)
	tracks := []model.Track{}
	cached := false
	if raw, hit, err := server.cache.Get(request.Context(), cacheKey); err == nil && hit && json.Unmarshal(raw, &tracks) == nil {
		cached = true
		server.cacheHits.Add(1)
	} else {
		server.cacheMisses.Add(1)
		var err error
		tracks, err = server.catalog.Recommend(input.Emotion, input.ContextKey, input.Limit)
		if errors.Is(err, catalog.ErrUnsupportedEmotion) {
			writeError(writer, http.StatusUnprocessableEntity, "emotion must be one of happy, sad, angry, calm, fear, or love")
			return
		}
		if err != nil {
			server.internalError(writer, "recommend tracks", err)
			return
		}
		if server.spotify.Enabled() {
			enrichmentContext, cancelEnrichment := context.WithTimeout(request.Context(), 3*time.Second)
			for index, track := range tracks {
				if enrichmentContext.Err() != nil {
					break
				}
				enriched, enrichErr := server.spotify.Enrich(enrichmentContext, track)
				if enrichErr != nil {
					server.logger.Warn("spotify enrichment unavailable", "track_id", track.ID, "error", enrichErr)
					continue
				}
				tracks[index] = enriched
			}
			cancelEnrichment()
		}
		if raw, err := json.Marshal(tracks); err == nil {
			if err := server.cache.Set(request.Context(), cacheKey, raw, 15*time.Minute); err != nil {
				server.logger.Warn("cache recommendation feed", "error", err)
			}
		}
	}

	recommendation := model.Recommendation{
		ID: newID("rec"), ProfileID: input.ProfileID, Emotion: input.Emotion,
		Confidence: input.Confidence, Tracks: tracks, Cached: cached, CreatedAt: time.Now().UTC(),
	}
	if err := server.store.SaveRecommendation(request.Context(), recommendation); err != nil {
		server.internalError(writer, "save recommendation history", err)
		return
	}
	writeJSON(writer, http.StatusOK, recommendation)
}

func (server *Server) listHistory(writer http.ResponseWriter, request *http.Request) {
	history, err := server.store.ListHistory(request.Context(), request.PathValue("profileID"), queryLimit(request, 20))
	if err != nil {
		server.internalError(writer, "list recommendation history", err)
		return
	}
	writeJSON(writer, http.StatusOK, map[string]any{"history": history})
}

func (server *Server) createPlaylist(writer http.ResponseWriter, request *http.Request) {
	var input struct {
		ProfileID string `json:"profile_id"`
		Name      string `json:"name"`
	}
	if !decodeJSON(writer, request, &input) {
		return
	}
	input.Name = strings.TrimSpace(input.Name)
	if len(input.Name) < 1 || len(input.Name) > 80 {
		writeError(writer, http.StatusUnprocessableEntity, "name must contain 1 to 80 characters")
		return
	}
	if _, err := server.store.GetProfile(request.Context(), input.ProfileID); errors.Is(err, store.ErrNotFound) {
		writeError(writer, http.StatusNotFound, "profile not found")
		return
	} else if err != nil {
		server.internalError(writer, "validate profile", err)
		return
	}
	playlist, err := server.store.CreatePlaylist(request.Context(), input.ProfileID, input.Name)
	if err != nil {
		server.internalError(writer, "create playlist", err)
		return
	}
	writeJSON(writer, http.StatusCreated, playlist)
}

func (server *Server) listPlaylists(writer http.ResponseWriter, request *http.Request) {
	playlists, err := server.store.ListPlaylists(request.Context(), request.PathValue("profileID"))
	if err != nil {
		server.internalError(writer, "list playlists", err)
		return
	}
	writeJSON(writer, http.StatusOK, map[string]any{"playlists": playlists})
}

func (server *Server) getPlaylist(writer http.ResponseWriter, request *http.Request) {
	playlist, err := server.store.GetPlaylist(request.Context(), request.PathValue("playlistID"))
	if errors.Is(err, store.ErrNotFound) {
		writeError(writer, http.StatusNotFound, "playlist not found")
		return
	}
	if err != nil {
		server.internalError(writer, "get playlist", err)
		return
	}
	writeJSON(writer, http.StatusOK, playlist)
}

func (server *Server) addTrack(writer http.ResponseWriter, request *http.Request) {
	var input struct {
		TrackID string `json:"track_id"`
	}
	if !decodeJSON(writer, request, &input) {
		return
	}
	track, ok := server.catalog.Get(strings.TrimSpace(input.TrackID))
	if !ok {
		writeError(writer, http.StatusNotFound, "track not found")
		return
	}
	playlist, err := server.store.AddTrack(request.Context(), request.PathValue("playlistID"), track)
	if errors.Is(err, store.ErrNotFound) {
		writeError(writer, http.StatusNotFound, "playlist not found")
		return
	}
	if err != nil {
		server.internalError(writer, "add playlist track", err)
		return
	}
	writeJSON(writer, http.StatusOK, playlist)
}

func decodeJSON(writer http.ResponseWriter, request *http.Request, destination any) bool {
	decoder := json.NewDecoder(request.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		writeError(writer, http.StatusBadRequest, "request body must be valid JSON: "+err.Error())
		return false
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		writeError(writer, http.StatusBadRequest, "request body must contain a single JSON object")
		return false
	}
	return true
}

func queryLimit(request *http.Request, fallback int) int {
	limit, err := strconv.Atoi(request.URL.Query().Get("limit"))
	if err != nil || limit < 1 || limit > 100 {
		return fallback
	}
	return limit
}

func recommendationCacheKey(emotion, contextText string, limit int) string {
	digest := sha256.Sum256([]byte(strings.ToLower(strings.TrimSpace(emotion)) + "\x00" + strings.TrimSpace(contextText) + "\x00" + strconv.Itoa(limit)))
	return "recommendation:v1:" + hex.EncodeToString(digest[:])
}

func newID(prefix string) string {
	random := make([]byte, 12)
	if _, err := rand.Read(random); err != nil {
		return fmt.Sprintf("%s_%d", prefix, time.Now().UnixNano())
	}
	return prefix + "_" + hex.EncodeToString(random)
}

func writeJSON(writer http.ResponseWriter, status int, value any) {
	writer.WriteHeader(status)
	_ = json.NewEncoder(writer).Encode(value)
}

func writeError(writer http.ResponseWriter, status int, message string) {
	writeJSON(writer, status, map[string]any{"error": map[string]string{"message": message}})
}

func (server *Server) internalError(writer http.ResponseWriter, operation string, err error) {
	server.logger.Error(operation, "error", err)
	writeError(writer, http.StatusInternalServerError, "internal server error")
}

func Shutdown(ctx context.Context, httpServer *http.Server) error {
	return httpServer.Shutdown(ctx)
}
