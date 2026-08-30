package catalog

import (
	"crypto/sha256"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"math"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/akshatkachroo/spotisense/internal/model"
)

var ErrUnsupportedEmotion = errors.New("unsupported emotion")

var emotionTargets = map[string][5]float64{
	"happy": {0.86, 0.78, 0.82, 0.20, 0.05},
	"sad":   {0.18, 0.24, 0.25, 0.76, 0.12},
	"angry": {0.16, 0.90, 0.62, 0.14, 0.04},
	"calm":  {0.58, 0.24, 0.30, 0.82, 0.34},
	"fear":  {0.22, 0.58, 0.34, 0.44, 0.20},
	"love":  {0.80, 0.48, 0.58, 0.54, 0.07},
}

var featureWeights = [5]float64{0.32, 0.28, 0.18, 0.16, 0.06}

type Catalog struct {
	tracks []model.Track
	byID   map[string]model.Track
}

func Load(path string) (*Catalog, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open track catalog: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("read track catalog header: %w", err)
	}
	columns := make(map[string]int, len(header))
	for index, name := range header {
		columns[strings.TrimSpace(name)] = index
	}

	required := []string{"id", "name", "artist", "primary_emotion", "valence", "energy", "danceability", "acousticness", "instrumentalness", "data_source"}
	for _, name := range required {
		if _, ok := columns[name]; !ok {
			return nil, fmt.Errorf("track catalog is missing %q column", name)
		}
	}

	tracks := make([]model.Track, 0, 512)
	byID := make(map[string]model.Track, 512)
	for {
		record, readErr := reader.Read()
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return nil, fmt.Errorf("read track catalog row %d: %w", len(tracks)+2, readErr)
		}

		track, parseErr := parseTrack(record, columns)
		if parseErr != nil {
			return nil, fmt.Errorf("parse track catalog row %d: %w", len(tracks)+2, parseErr)
		}
		if _, duplicate := byID[track.ID]; duplicate {
			return nil, fmt.Errorf("duplicate track id %q", track.ID)
		}
		tracks = append(tracks, track)
		byID[track.ID] = track
	}

	if len(tracks) == 0 {
		return nil, errors.New("track catalog is empty")
	}
	return &Catalog{tracks: tracks, byID: byID}, nil
}

func (catalog *Catalog) Count() int {
	return len(catalog.tracks)
}

func (catalog *Catalog) Get(trackID string) (model.Track, bool) {
	track, ok := catalog.byID[trackID]
	return track, ok
}

func (catalog *Catalog) Search(query string, limit int) []model.Track {
	if limit < 1 || limit > 50 {
		limit = 10
	}
	needle := strings.ToLower(strings.TrimSpace(query))
	results := make([]model.Track, 0, limit)
	for _, track := range catalog.tracks {
		if needle == "" || strings.Contains(strings.ToLower(track.Name+" "+track.Artist), needle) {
			results = append(results, track)
			if len(results) == limit {
				break
			}
		}
	}
	return results
}

func (catalog *Catalog) Recommend(emotion, seed string, limit int) ([]model.Track, error) {
	emotion = strings.ToLower(strings.TrimSpace(emotion))
	target, ok := emotionTargets[emotion]
	if !ok {
		return nil, ErrUnsupportedEmotion
	}
	if limit < 1 || limit > 20 {
		limit = 5
	}

	type candidate struct {
		track model.Track
		score float64
	}
	candidates := make([]candidate, 0, len(catalog.tracks))
	for _, track := range catalog.tracks {
		features := [5]float64{track.Valence, track.Energy, track.Danceability, track.Acousticness, track.Instrumentalness}
		distance := 0.0
		for index, value := range features {
			delta := value - target[index]
			distance += featureWeights[index] * delta * delta
		}
		score := 1 - math.Sqrt(distance)
		if track.PrimaryEmotion == emotion {
			score += 0.06
		}
		score += stableJitter(seed, track.ID)
		candidates = append(candidates, candidate{track: track, score: clamp(score, 0, 1)})
	}

	sort.SliceStable(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	results := make([]model.Track, 0, limit)
	artists := make(map[string]struct{}, limit)
	for _, candidate := range candidates {
		artistKey := strings.ToLower(candidate.track.Artist)
		if _, alreadySelected := artists[artistKey]; alreadySelected {
			continue
		}
		candidate.track.MatchScore = math.Round(candidate.score*1000) / 1000
		results = append(results, candidate.track)
		artists[artistKey] = struct{}{}
		if len(results) == limit {
			break
		}
	}
	return results, nil
}

func parseTrack(record []string, columns map[string]int) (model.Track, error) {
	value := func(name string) string {
		index, ok := columns[name]
		if !ok || index >= len(record) {
			return ""
		}
		return strings.TrimSpace(record[index])
	}
	parseFeature := func(name string) (float64, error) {
		parsed, err := strconv.ParseFloat(value(name), 64)
		if err != nil || parsed < 0 || parsed > 1 {
			return 0, fmt.Errorf("%s must be a number between 0 and 1", name)
		}
		return parsed, nil
	}

	track := model.Track{
		ID:             value("id"),
		Name:           value("name"),
		Artist:         value("artist"),
		PrimaryEmotion: value("primary_emotion"),
		SpotifyURL:     value("spotify_url"),
		DataSource:     value("data_source"),
	}
	if track.ID == "" || track.Name == "" || track.Artist == "" {
		return model.Track{}, errors.New("id, name, and artist are required")
	}
	if track.SpotifyURL == "" {
		track.SpotifyURL = spotifySearchURL(track.Name, track.Artist)
	}

	var err error
	if track.Valence, err = parseFeature("valence"); err != nil {
		return model.Track{}, err
	}
	if track.Energy, err = parseFeature("energy"); err != nil {
		return model.Track{}, err
	}
	if track.Danceability, err = parseFeature("danceability"); err != nil {
		return model.Track{}, err
	}
	if track.Acousticness, err = parseFeature("acousticness"); err != nil {
		return model.Track{}, err
	}
	if track.Instrumentalness, err = parseFeature("instrumentalness"); err != nil {
		return model.Track{}, err
	}
	return track, nil
}

func stableJitter(seed, trackID string) float64 {
	digest := sha256.Sum256([]byte(strings.TrimSpace(seed) + ":" + trackID))
	normalized := float64(digest[0])/255.0 - 0.5
	return normalized * 0.035
}

func spotifySearchURL(name, artist string) string {
	return "https://open.spotify.com/search/" + url.PathEscape(strings.TrimSpace(name+" "+artist))
}

func clamp(value, minimum, maximum float64) float64 {
	return math.Max(minimum, math.Min(maximum, value))
}
