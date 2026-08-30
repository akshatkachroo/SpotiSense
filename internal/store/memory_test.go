package store

import (
	"context"
	"testing"

	"github.com/akshatkachroo/spotisense/internal/model"
)

func TestMemoryPlaylistIsIdempotent(t *testing.T) {
	ctx := context.Background()
	memory := NewMemory()
	profile, err := memory.CreateProfile(ctx, "Listener")
	if err != nil {
		t.Fatal(err)
	}
	playlist, err := memory.CreatePlaylist(ctx, profile.ID, "Focus")
	if err != nil {
		t.Fatal(err)
	}
	track := model.Track{ID: "track-one", Name: "One", Artist: "Artist"}
	if _, err := memory.AddTrack(ctx, playlist.ID, track); err != nil {
		t.Fatal(err)
	}
	updated, err := memory.AddTrack(ctx, playlist.ID, track)
	if err != nil {
		t.Fatal(err)
	}
	if len(updated.Tracks) != 1 {
		t.Fatalf("expected one unique track, got %d", len(updated.Tracks))
	}
}
