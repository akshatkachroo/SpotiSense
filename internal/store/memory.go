package store

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/akshatkachroo/spotisense/internal/model"
)

type Memory struct {
	mutex           sync.RWMutex
	profiles        map[string]model.Profile
	recommendations map[string][]model.Recommendation
	playlists       map[string]model.Playlist
}

func NewMemory() *Memory {
	return &Memory{
		profiles:        make(map[string]model.Profile),
		recommendations: make(map[string][]model.Recommendation),
		playlists:       make(map[string]model.Playlist),
	}
}

func (memory *Memory) CreateProfile(_ context.Context, displayName string) (model.Profile, error) {
	profile := model.Profile{ID: newID("pro"), DisplayName: strings.TrimSpace(displayName), CreatedAt: time.Now().UTC()}
	memory.mutex.Lock()
	memory.profiles[profile.ID] = profile
	memory.mutex.Unlock()
	return profile, nil
}

func (memory *Memory) GetProfile(_ context.Context, profileID string) (model.Profile, error) {
	memory.mutex.RLock()
	profile, ok := memory.profiles[profileID]
	memory.mutex.RUnlock()
	if !ok {
		return model.Profile{}, ErrNotFound
	}
	return profile, nil
}

func (memory *Memory) SaveRecommendation(_ context.Context, recommendation model.Recommendation) error {
	memory.mutex.Lock()
	memory.recommendations[recommendation.ProfileID] = append(memory.recommendations[recommendation.ProfileID], recommendation)
	memory.mutex.Unlock()
	return nil
}

func (memory *Memory) ListHistory(_ context.Context, profileID string, limit int) ([]model.Recommendation, error) {
	memory.mutex.RLock()
	source := memory.recommendations[profileID]
	memory.mutex.RUnlock()
	if limit < 1 || limit > 100 {
		limit = 20
	}
	start := len(source) - limit
	if start < 0 {
		start = 0
	}
	result := make([]model.Recommendation, 0, len(source)-start)
	for index := len(source) - 1; index >= start; index-- {
		result = append(result, source[index])
	}
	return result, nil
}

func (memory *Memory) CreatePlaylist(_ context.Context, profileID, name string) (model.Playlist, error) {
	now := time.Now().UTC()
	playlist := model.Playlist{ID: newID("ply"), ProfileID: profileID, Name: strings.TrimSpace(name), Tracks: []model.Track{}, CreatedAt: now, UpdatedAt: now}
	memory.mutex.Lock()
	memory.playlists[playlist.ID] = playlist
	memory.mutex.Unlock()
	return playlist, nil
}

func (memory *Memory) ListPlaylists(_ context.Context, profileID string) ([]model.Playlist, error) {
	memory.mutex.RLock()
	result := make([]model.Playlist, 0)
	for _, playlist := range memory.playlists {
		if playlist.ProfileID == profileID {
			result = append(result, clonePlaylist(playlist))
		}
	}
	memory.mutex.RUnlock()
	sort.Slice(result, func(i, j int) bool { return result[i].UpdatedAt.After(result[j].UpdatedAt) })
	return result, nil
}

func (memory *Memory) GetPlaylist(_ context.Context, playlistID string) (model.Playlist, error) {
	memory.mutex.RLock()
	playlist, ok := memory.playlists[playlistID]
	memory.mutex.RUnlock()
	if !ok {
		return model.Playlist{}, ErrNotFound
	}
	return clonePlaylist(playlist), nil
}

func (memory *Memory) AddTrack(_ context.Context, playlistID string, track model.Track) (model.Playlist, error) {
	memory.mutex.Lock()
	defer memory.mutex.Unlock()
	playlist, ok := memory.playlists[playlistID]
	if !ok {
		return model.Playlist{}, ErrNotFound
	}
	for _, existing := range playlist.Tracks {
		if existing.ID == track.ID {
			return clonePlaylist(playlist), nil
		}
	}
	playlist.Tracks = append(playlist.Tracks, track)
	playlist.UpdatedAt = time.Now().UTC()
	memory.playlists[playlistID] = playlist
	return clonePlaylist(playlist), nil
}

func (memory *Memory) Backend() string { return "memory" }
func (memory *Memory) Close()          {}

func newID(prefix string) string {
	buffer := make([]byte, 12)
	if _, err := rand.Read(buffer); err != nil {
		return prefix + "_" + time.Now().UTC().Format("20060102150405.000000000")
	}
	return prefix + "_" + hex.EncodeToString(buffer)
}

func clonePlaylist(playlist model.Playlist) model.Playlist {
	playlist.Tracks = append([]model.Track(nil), playlist.Tracks...)
	return playlist
}
