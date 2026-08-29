package spotify

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/akshatkachroo/spotisense/internal/cache"
	"github.com/akshatkachroo/spotisense/internal/model"
)

const (
	accountsEndpoint = "https://accounts.spotify.com/api/token"
	searchEndpoint   = "https://api.spotify.com/v1/search"
)

type Client struct {
	clientID     string
	clientSecret string
	httpClient   *http.Client
	cache        cache.Cache
	tokenMutex   sync.Mutex
	token        string
	tokenExpiry  time.Time
}

func New(clientID, clientSecret string, responseCache cache.Cache) *Client {
	return &Client{
		clientID:     strings.TrimSpace(clientID),
		clientSecret: strings.TrimSpace(clientSecret),
		httpClient:   &http.Client{Timeout: 6 * time.Second},
		cache:        responseCache,
	}
}

func (client *Client) Enabled() bool {
	return client.clientID != "" && client.clientSecret != ""
}

func (client *Client) Enrich(ctx context.Context, track model.Track) (model.Track, error) {
	if !client.Enabled() {
		return track, nil
	}
	key := spotifyCacheKey(track.Name, track.Artist)
	if raw, hit, err := client.cache.Get(ctx, key); err == nil && hit {
		var cached model.Track
		if json.Unmarshal(raw, &cached) == nil {
			track.ImageURL = cached.ImageURL
			track.SpotifyURL = cached.SpotifyURL
			return track, nil
		}
	}

	token, err := client.accessToken(ctx)
	if err != nil {
		return track, err
	}
	query := url.Values{}
	query.Set("q", fmt.Sprintf("track:%q artist:%q", track.Name, track.Artist))
	query.Set("type", "track")
	query.Set("limit", "1")
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, searchEndpoint+"?"+query.Encode(), nil)
	if err != nil {
		return track, err
	}
	request.Header.Set("Authorization", "Bearer "+token)

	response, err := client.httpClient.Do(request)
	if err != nil {
		return track, err
	}
	defer response.Body.Close()
	if response.StatusCode == http.StatusTooManyRequests {
		return track, fmt.Errorf("spotify quota exceeded; retry after %s seconds", response.Header.Get("Retry-After"))
	}
	if response.StatusCode != http.StatusOK {
		return track, fmt.Errorf("spotify search returned %s", response.Status)
	}

	var payload struct {
		Tracks struct {
			Items []struct {
				ExternalURLs map[string]string `json:"external_urls"`
				Album        struct {
					Images []struct {
						URL string `json:"url"`
					} `json:"images"`
				} `json:"album"`
			} `json:"items"`
		} `json:"tracks"`
	}
	if err := json.NewDecoder(response.Body).Decode(&payload); err != nil {
		return track, err
	}
	if len(payload.Tracks.Items) == 0 {
		return track, errors.New("spotify track not found")
	}
	item := payload.Tracks.Items[0]
	if spotifyURL := item.ExternalURLs["spotify"]; spotifyURL != "" {
		track.SpotifyURL = spotifyURL
	}
	if len(item.Album.Images) > 0 {
		track.ImageURL = item.Album.Images[0].URL
	}
	if raw, err := json.Marshal(track); err == nil {
		_ = client.cache.Set(ctx, key, raw, 24*time.Hour)
	}
	return track, nil
}

func (client *Client) accessToken(ctx context.Context) (string, error) {
	client.tokenMutex.Lock()
	defer client.tokenMutex.Unlock()
	if client.token != "" && time.Now().Add(30*time.Second).Before(client.tokenExpiry) {
		return client.token, nil
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, accountsEndpoint, strings.NewReader("grant_type=client_credentials"))
	if err != nil {
		return "", err
	}
	request.SetBasicAuth(client.clientID, client.clientSecret)
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	response, err := client.httpClient.Do(request)
	if err != nil {
		return "", err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return "", fmt.Errorf("spotify token endpoint returned %s", response.Status)
	}
	var payload struct {
		AccessToken string `json:"access_token"`
		ExpiresIn   int    `json:"expires_in"`
	}
	if err := json.NewDecoder(response.Body).Decode(&payload); err != nil {
		return "", err
	}
	if payload.AccessToken == "" {
		return "", errors.New("spotify returned an empty access token")
	}
	client.token = payload.AccessToken
	client.tokenExpiry = time.Now().Add(time.Duration(payload.ExpiresIn) * time.Second)
	return client.token, nil
}

func spotifyCacheKey(name, artist string) string {
	digest := sha256.Sum256([]byte(strings.ToLower(strings.TrimSpace(name + "\x00" + artist))))
	return "spotify:track:" + hex.EncodeToString(digest[:])
}
