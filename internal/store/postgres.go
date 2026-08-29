package store

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/akshatkachroo/spotisense/internal/model"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

//go:embed schema.sql
var schema string

type Postgres struct {
	pool *pgxpool.Pool
}

func NewPostgres(ctx context.Context, connectionURL string) (*Postgres, error) {
	configuration, err := pgxpool.ParseConfig(connectionURL)
	if err != nil {
		return nil, err
	}
	configuration.MaxConns = 5
	configuration.MinConns = 0
	configuration.MaxConnIdleTime = 2 * time.Minute
	pool, err := pgxpool.NewWithConfig(ctx, configuration)
	if err != nil {
		return nil, err
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, err
	}
	if _, err := pool.Exec(ctx, schema); err != nil {
		pool.Close()
		return nil, fmt.Errorf("apply database schema: %w", err)
	}
	return &Postgres{pool: pool}, nil
}

func (postgres *Postgres) CreateProfile(ctx context.Context, displayName string) (model.Profile, error) {
	profile := model.Profile{ID: newID("pro"), DisplayName: strings.TrimSpace(displayName), CreatedAt: time.Now().UTC()}
	_, err := postgres.pool.Exec(ctx, `INSERT INTO profiles (id, display_name, created_at) VALUES ($1, $2, $3)`, profile.ID, profile.DisplayName, profile.CreatedAt)
	return profile, err
}

func (postgres *Postgres) GetProfile(ctx context.Context, profileID string) (model.Profile, error) {
	var profile model.Profile
	err := postgres.pool.QueryRow(ctx, `SELECT id, display_name, created_at FROM profiles WHERE id = $1`, profileID).Scan(&profile.ID, &profile.DisplayName, &profile.CreatedAt)
	return profile, translateNotFound(err)
}

func (postgres *Postgres) SaveRecommendation(ctx context.Context, recommendation model.Recommendation) error {
	tracks, err := json.Marshal(recommendation.Tracks)
	if err != nil {
		return err
	}
	_, err = postgres.pool.Exec(ctx, `
		INSERT INTO recommendation_history (id, profile_id, emotion, confidence, tracks, cached, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7)`,
		recommendation.ID, recommendation.ProfileID, recommendation.Emotion, recommendation.Confidence, tracks, recommendation.Cached, recommendation.CreatedAt,
	)
	return err
}

func (postgres *Postgres) ListHistory(ctx context.Context, profileID string, limit int) ([]model.Recommendation, error) {
	if limit < 1 || limit > 100 {
		limit = 20
	}
	rows, err := postgres.pool.Query(ctx, `
		SELECT id, profile_id, emotion, confidence, tracks, cached, created_at
		FROM recommendation_history WHERE profile_id = $1 ORDER BY created_at DESC LIMIT $2`, profileID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	result := make([]model.Recommendation, 0, limit)
	for rows.Next() {
		var recommendation model.Recommendation
		var tracks []byte
		if err := rows.Scan(&recommendation.ID, &recommendation.ProfileID, &recommendation.Emotion, &recommendation.Confidence, &tracks, &recommendation.Cached, &recommendation.CreatedAt); err != nil {
			return nil, err
		}
		if err := json.Unmarshal(tracks, &recommendation.Tracks); err != nil {
			return nil, err
		}
		result = append(result, recommendation)
	}
	return result, rows.Err()
}

func (postgres *Postgres) CreatePlaylist(ctx context.Context, profileID, name string) (model.Playlist, error) {
	now := time.Now().UTC()
	playlist := model.Playlist{ID: newID("ply"), ProfileID: profileID, Name: strings.TrimSpace(name), Tracks: []model.Track{}, CreatedAt: now, UpdatedAt: now}
	_, err := postgres.pool.Exec(ctx, `INSERT INTO playlists (id, profile_id, name, created_at, updated_at) VALUES ($1, $2, $3, $4, $5)`, playlist.ID, playlist.ProfileID, playlist.Name, playlist.CreatedAt, playlist.UpdatedAt)
	return playlist, err
}

func (postgres *Postgres) ListPlaylists(ctx context.Context, profileID string) ([]model.Playlist, error) {
	rows, err := postgres.pool.Query(ctx, `SELECT id FROM playlists WHERE profile_id = $1 ORDER BY updated_at DESC`, profileID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	ids := make([]string, 0)
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	result := make([]model.Playlist, 0, len(ids))
	for _, id := range ids {
		playlist, err := postgres.GetPlaylist(ctx, id)
		if err != nil {
			return nil, err
		}
		result = append(result, playlist)
	}
	return result, nil
}

func (postgres *Postgres) GetPlaylist(ctx context.Context, playlistID string) (model.Playlist, error) {
	var playlist model.Playlist
	err := postgres.pool.QueryRow(ctx, `SELECT id, profile_id, name, created_at, updated_at FROM playlists WHERE id = $1`, playlistID).Scan(
		&playlist.ID, &playlist.ProfileID, &playlist.Name, &playlist.CreatedAt, &playlist.UpdatedAt,
	)
	if err != nil {
		return model.Playlist{}, translateNotFound(err)
	}
	playlist.Tracks = []model.Track{}
	rows, err := postgres.pool.Query(ctx, `SELECT track FROM playlist_tracks WHERE playlist_id = $1 ORDER BY position`, playlistID)
	if err != nil {
		return model.Playlist{}, err
	}
	defer rows.Close()
	for rows.Next() {
		var raw []byte
		var track model.Track
		if err := rows.Scan(&raw); err != nil {
			return model.Playlist{}, err
		}
		if err := json.Unmarshal(raw, &track); err != nil {
			return model.Playlist{}, err
		}
		playlist.Tracks = append(playlist.Tracks, track)
	}
	return playlist, rows.Err()
}

func (postgres *Postgres) AddTrack(ctx context.Context, playlistID string, track model.Track) (model.Playlist, error) {
	if _, err := postgres.GetPlaylist(ctx, playlistID); err != nil {
		return model.Playlist{}, err
	}
	raw, err := json.Marshal(track)
	if err != nil {
		return model.Playlist{}, err
	}
	tx, err := postgres.pool.Begin(ctx)
	if err != nil {
		return model.Playlist{}, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	result, err := tx.Exec(ctx, `INSERT INTO playlist_tracks (playlist_id, track_id, track) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING`, playlistID, track.ID, raw)
	if err != nil {
		return model.Playlist{}, translateNotFound(err)
	}
	if result.RowsAffected() > 0 {
		if _, err := tx.Exec(ctx, `UPDATE playlists SET updated_at = now() WHERE id = $1`, playlistID); err != nil {
			return model.Playlist{}, err
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return model.Playlist{}, err
	}
	return postgres.GetPlaylist(ctx, playlistID)
}

func (postgres *Postgres) Backend() string { return "postgres" }
func (postgres *Postgres) Close()          { postgres.pool.Close() }

func translateNotFound(err error) error {
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrNotFound
	}
	return err
}
