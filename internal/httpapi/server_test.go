package httpapi

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/akshatkachroo/spotisense/internal/cache"
	"github.com/akshatkachroo/spotisense/internal/catalog"
	"github.com/akshatkachroo/spotisense/internal/spotify"
	"github.com/akshatkachroo/spotisense/internal/store"
)

const apiTestCatalog = `id,name,artist,primary_emotion,valence,energy,danceability,acousticness,instrumentalness,spotify_url,data_source
one,Sunrise,Artist A,happy,0.90,0.80,0.80,0.10,0.05,https://example.com/one,test
two,Good Day,Artist B,happy,0.85,0.75,0.85,0.20,0.05,https://example.com/two,test
three,Celebrate,Artist C,happy,0.88,0.85,0.75,0.12,0.02,https://example.com/three,test
four,Slow Rain,Artist D,sad,0.10,0.20,0.20,0.90,0.10,https://example.com/four,test
`

func testHandler(t *testing.T) http.Handler {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tracks.csv")
	if err := os.WriteFile(path, []byte(apiTestCatalog), 0o600); err != nil {
		t.Fatal(err)
	}
	trackCatalog, err := catalog.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	responseCache := cache.NewMemory()
	dataStore := store.NewMemory()
	spotifyClient := spotify.New("", "", responseCache)
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	return New(dataStore, responseCache, trackCatalog, spotifyClient, logger, "https://spotisense.example").Handler()
}

func requestJSON(t *testing.T, handler http.Handler, method, path string, body any, status int) map[string]any {
	t.Helper()
	var reader io.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
		reader = bytes.NewReader(raw)
	}
	request := httptest.NewRequest(method, path, reader)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != status {
		t.Fatalf("%s %s returned %d, want %d: %s", method, path, response.Code, status, response.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	return payload
}

func TestRecommendationWorkflowCachesAndPersistsHistory(t *testing.T) {
	handler := testHandler(t)
	profile := requestJSON(t, handler, http.MethodPost, "/v1/profiles", map[string]any{"display_name": "Listener"}, http.StatusCreated)
	profileID := profile["id"].(string)
	body := map[string]any{"profile_id": profileID, "emotion": "happy", "confidence": 0.91, "context_key": "3d52f9d9089e38c9a04f", "limit": 3}
	first := requestJSON(t, handler, http.MethodPost, "/v1/recommendations", body, http.StatusOK)
	second := requestJSON(t, handler, http.MethodPost, "/v1/recommendations", body, http.StatusOK)
	if first["cached"].(bool) || !second["cached"].(bool) {
		t.Fatalf("expected miss followed by hit: first=%v second=%v", first["cached"], second["cached"])
	}
	history := requestJSON(t, handler, http.MethodGet, "/v1/profiles/"+profileID+"/history", nil, http.StatusOK)
	if len(history["history"].([]any)) != 2 {
		t.Fatalf("expected two history records: %#v", history)
	}
}

func TestPlaylistWorkflow(t *testing.T) {
	handler := testHandler(t)
	profile := requestJSON(t, handler, http.MethodPost, "/v1/profiles", map[string]any{"display_name": "Listener"}, http.StatusCreated)
	playlist := requestJSON(t, handler, http.MethodPost, "/v1/playlists", map[string]any{"profile_id": profile["id"], "name": "Focus"}, http.StatusCreated)
	updated := requestJSON(t, handler, http.MethodPost, "/v1/playlists/"+playlist["id"].(string)+"/tracks", map[string]any{"track_id": "one"}, http.StatusOK)
	if len(updated["tracks"].([]any)) != 1 {
		t.Fatalf("expected saved track: %#v", updated)
	}
}

func TestCORSPreflightAllowsConfiguredFrontend(t *testing.T) {
	handler := testHandler(t)
	request := httptest.NewRequest(http.MethodOptions, "/v1/recommendations", nil)
	request.Header.Set("Origin", "https://spotisense.example")
	request.Header.Set("Access-Control-Request-Method", http.MethodPost)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)

	if response.Code != http.StatusNoContent {
		t.Fatalf("preflight returned %d, want %d", response.Code, http.StatusNoContent)
	}
	if origin := response.Header().Get("Access-Control-Allow-Origin"); origin != "https://spotisense.example" {
		t.Fatalf("allow origin = %q", origin)
	}
}

func TestCORSDoesNotAllowUnknownOrigin(t *testing.T) {
	handler := testHandler(t)
	request := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	request.Header.Set("Origin", "https://untrusted.example")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)

	if origin := response.Header().Get("Access-Control-Allow-Origin"); origin != "" {
		t.Fatalf("unexpected allow origin %q", origin)
	}
}
