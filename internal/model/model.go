package model

import "time"

type Profile struct {
	ID          string    `json:"id"`
	DisplayName string    `json:"display_name"`
	CreatedAt   time.Time `json:"created_at"`
}

type Track struct {
	ID               string  `json:"id"`
	Name             string  `json:"name"`
	Artist           string  `json:"artist"`
	PrimaryEmotion   string  `json:"primary_emotion"`
	Valence          float64 `json:"valence"`
	Energy           float64 `json:"energy"`
	Danceability     float64 `json:"danceability"`
	Acousticness     float64 `json:"acousticness"`
	Instrumentalness float64 `json:"instrumentalness"`
	MatchScore       float64 `json:"match_score,omitempty"`
	SpotifyURL       string  `json:"spotify_url"`
	ImageURL         string  `json:"image_url,omitempty"`
	DataSource       string  `json:"data_source"`
}

type Recommendation struct {
	ID         string    `json:"id"`
	ProfileID  string    `json:"profile_id,omitempty"`
	Emotion    string    `json:"emotion"`
	Confidence float64   `json:"confidence"`
	Tracks     []Track   `json:"tracks"`
	Cached     bool      `json:"cached"`
	CreatedAt  time.Time `json:"created_at"`
}

type Playlist struct {
	ID        string    `json:"id"`
	ProfileID string    `json:"profile_id"`
	Name      string    `json:"name"`
	Tracks    []Track   `json:"tracks"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}
