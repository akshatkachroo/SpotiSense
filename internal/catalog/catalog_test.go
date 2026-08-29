package catalog

import (
	"os"
	"path/filepath"
	"testing"
)

const testCatalog = `id,name,artist,primary_emotion,valence,energy,danceability,acousticness,instrumentalness,spotify_url,data_source
one,Sunrise,Artist A,happy,0.90,0.80,0.80,0.10,0.05,https://example.com/one,test
two,Slow Rain,Artist B,sad,0.10,0.20,0.20,0.90,0.10,https://example.com/two,test
three,Dance Again,Artist C,happy,0.85,0.75,0.90,0.10,0.01,https://example.com/three,test
`

func loadTestCatalog(t *testing.T) *Catalog {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tracks.csv")
	if err := os.WriteFile(path, []byte(testCatalog), 0o600); err != nil {
		t.Fatal(err)
	}
	catalog, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	return catalog
}

func TestRecommendRanksTargetMoodAndIsStable(t *testing.T) {
	catalog := loadTestCatalog(t)
	first, err := catalog.Recommend("happy", "great day", 2)
	if err != nil {
		t.Fatal(err)
	}
	second, err := catalog.Recommend("happy", "great day", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != 2 || first[0].PrimaryEmotion != "happy" {
		t.Fatalf("unexpected recommendations: %#v", first)
	}
	if first[0].ID != second[0].ID || first[0].MatchScore != second[0].MatchScore {
		t.Fatalf("same seed should produce stable results: %#v vs %#v", first, second)
	}
}

func TestSearchMatchesArtistAndTitle(t *testing.T) {
	catalog := loadTestCatalog(t)
	if results := catalog.Search("rain", 10); len(results) != 1 || results[0].ID != "two" {
		t.Fatalf("unexpected search results: %#v", results)
	}
}
