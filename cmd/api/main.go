package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/akshatkachroo/spotisense/internal/cache"
	"github.com/akshatkachroo/spotisense/internal/catalog"
	"github.com/akshatkachroo/spotisense/internal/config"
	"github.com/akshatkachroo/spotisense/internal/httpapi"
	"github.com/akshatkachroo/spotisense/internal/spotify"
	"github.com/akshatkachroo/spotisense/internal/store"
)

func main() {
	configuration := config.FromEnvironment()
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	trackCatalog, err := catalog.Load(configuration.TracksPath)
	if err != nil {
		logger.Error("load catalog", "error", err)
		os.Exit(1)
	}

	dataStore := initializeStore(ctx, configuration, logger)
	defer dataStore.Close()
	responseCache := initializeCache(ctx, configuration, logger)
	defer responseCache.Close()

	spotifyClient := spotify.New(configuration.SpotifyClientID, configuration.SpotifyClientSecret, responseCache)
	api := httpapi.New(dataStore, responseCache, trackCatalog, spotifyClient, logger, configuration.AllowedOrigins...)
	httpServer := &http.Server{
		Addr:              ":" + configuration.Port,
		Handler:           api.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      15 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	go func() {
		logger.Info("api started", "address", httpServer.Addr, "environment", configuration.Environment, "catalog_tracks", trackCatalog.Count(), "persistence", dataStore.Backend(), "cache", responseCache.Backend(), "spotify_enrichment", spotifyClient.Enabled())
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("serve api", "error", err)
			stop()
		}
	}()

	<-ctx.Done()
	shutdownContext, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := httpapi.Shutdown(shutdownContext, httpServer); err != nil {
		logger.Error("shutdown api", "error", err)
	}
}

func initializeStore(ctx context.Context, configuration config.Config, logger *slog.Logger) store.Store {
	if configuration.DatabaseURL == "" {
		logger.Warn("DATABASE_URL not set; profiles, playlists, and history will use in-memory storage")
		return store.NewMemory()
	}
	postgresStore, err := store.NewPostgres(ctx, configuration.DatabaseURL)
	if err != nil {
		logger.Error("connect to postgres", "error", err)
		os.Exit(1)
	}
	return postgresStore
}

func initializeCache(ctx context.Context, configuration config.Config, logger *slog.Logger) cache.Cache {
	if configuration.RedisURL == "" {
		logger.Warn("REDIS_URL not set; responses will use an in-memory cache")
		return cache.NewMemory()
	}
	redisCache, err := cache.NewRedis(ctx, configuration.RedisURL)
	if err != nil {
		logger.Error("connect to redis", "error", err)
		os.Exit(1)
	}
	return redisCache
}
