package store

import (
	"context"
	"errors"

	"github.com/akshatkachroo/spotisense/internal/model"
)

var ErrNotFound = errors.New("not found")

type Store interface {
	CreateProfile(context.Context, string) (model.Profile, error)
	GetProfile(context.Context, string) (model.Profile, error)
	SaveRecommendation(context.Context, model.Recommendation) error
	ListHistory(context.Context, string, int) ([]model.Recommendation, error)
	CreatePlaylist(context.Context, string, string) (model.Playlist, error)
	ListPlaylists(context.Context, string) ([]model.Playlist, error)
	GetPlaylist(context.Context, string) (model.Playlist, error)
	AddTrack(context.Context, string, model.Track) (model.Playlist, error)
	Backend() string
	Close()
}
