package config

import (
	"os"
	"strings"
)

type Config struct {
	Port                string
	Environment         string
	DatabaseURL         string
	RedisURL            string
	TracksPath          string
	AllowedOrigins      []string
	SpotifyClientID     string
	SpotifyClientSecret string
}

func FromEnvironment() Config {
	return Config{
		Port:                valueOrDefault("PORT", "8080"),
		Environment:         valueOrDefault("APP_ENV", "development"),
		DatabaseURL:         strings.TrimSpace(os.Getenv("DATABASE_URL")),
		RedisURL:            strings.TrimSpace(os.Getenv("REDIS_URL")),
		TracksPath:          valueOrDefault("TRACKS_PATH", "data/tracks.csv"),
		AllowedOrigins:      commaSeparated(valueOrDefault("CORS_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")),
		SpotifyClientID:     strings.TrimSpace(firstNonEmpty(os.Getenv("SPOTIFY_CLIENT_ID"), os.Getenv("CLIENT_ID"))),
		SpotifyClientSecret: strings.TrimSpace(firstNonEmpty(os.Getenv("SPOTIFY_CLIENT_SECRET"), os.Getenv("CLIENT_SECRET"))),
	}
}

func commaSeparated(value string) []string {
	parts := strings.Split(value, ",")
	values := make([]string, 0, len(parts))
	for _, part := range parts {
		if clean := strings.TrimSpace(part); clean != "" {
			values = append(values, strings.TrimRight(clean, "/"))
		}
	}
	return values
}

func valueOrDefault(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}
