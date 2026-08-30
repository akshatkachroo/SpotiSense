package cache

import (
	"context"
	"testing"
	"time"
)

func TestMemoryCacheStoresCopiesAndExpiresValues(t *testing.T) {
	cache := NewMemory()
	value := []byte("first")
	if err := cache.Set(context.Background(), "key", value, 10*time.Millisecond); err != nil {
		t.Fatal(err)
	}
	value[0] = 'x'
	cached, hit, err := cache.Get(context.Background(), "key")
	if err != nil || !hit || string(cached) != "first" {
		t.Fatalf("unexpected cache result: value=%q hit=%v err=%v", cached, hit, err)
	}
	time.Sleep(15 * time.Millisecond)
	if _, hit, err := cache.Get(context.Background(), "key"); err != nil || hit {
		t.Fatalf("expected expired cache miss, hit=%v err=%v", hit, err)
	}
}
