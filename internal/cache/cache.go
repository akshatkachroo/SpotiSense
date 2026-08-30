package cache

import (
	"context"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

type Cache interface {
	Get(context.Context, string) ([]byte, bool, error)
	Set(context.Context, string, []byte, time.Duration) error
	Backend() string
	Close() error
}

type memoryEntry struct {
	value     []byte
	expiresAt time.Time
}

type Memory struct {
	mutex   sync.RWMutex
	entries map[string]memoryEntry
}

func NewMemory() *Memory {
	return &Memory{entries: make(map[string]memoryEntry)}
}

func (memory *Memory) Get(_ context.Context, key string) ([]byte, bool, error) {
	memory.mutex.RLock()
	entry, ok := memory.entries[key]
	memory.mutex.RUnlock()
	if !ok {
		return nil, false, nil
	}
	if time.Now().After(entry.expiresAt) {
		memory.mutex.Lock()
		delete(memory.entries, key)
		memory.mutex.Unlock()
		return nil, false, nil
	}
	return append([]byte(nil), entry.value...), true, nil
}

func (memory *Memory) Set(_ context.Context, key string, value []byte, ttl time.Duration) error {
	memory.mutex.Lock()
	memory.entries[key] = memoryEntry{value: append([]byte(nil), value...), expiresAt: time.Now().Add(ttl)}
	memory.mutex.Unlock()
	return nil
}

func (memory *Memory) Backend() string { return "memory" }
func (memory *Memory) Close() error    { return nil }

type Redis struct {
	client *redis.Client
}

func NewRedis(ctx context.Context, connectionURL string) (*Redis, error) {
	options, err := redis.ParseURL(connectionURL)
	if err != nil {
		return nil, err
	}
	client := redis.NewClient(options)
	if err := client.Ping(ctx).Err(); err != nil {
		_ = client.Close()
		return nil, err
	}
	return &Redis{client: client}, nil
}

func (redisCache *Redis) Get(ctx context.Context, key string) ([]byte, bool, error) {
	value, err := redisCache.client.Get(ctx, key).Bytes()
	if err == redis.Nil {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, err
	}
	return value, true, nil
}

func (redisCache *Redis) Set(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	return redisCache.client.Set(ctx, key, value, ttl).Err()
}

func (redisCache *Redis) Backend() string { return "redis" }
func (redisCache *Redis) Close() error    { return redisCache.client.Close() }
