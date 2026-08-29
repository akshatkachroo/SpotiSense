.PHONY: setup setup-web test test-python test-go test-web lint dev-api dev-web demo catalog prepare-data train export-model

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements-dev.txt
	npm --prefix web install

setup-web:
	npm --prefix web install

test: test-python test-go test-web

test-python:
	.venv/bin/python -m pytest -q

test-go:
	go test ./cmd/... ./internal/...

test-web:
	npm --prefix web test
	npm --prefix web run build

lint:
	.venv/bin/python -m ruff check ml tests
	go vet ./cmd/... ./internal/...
	npm --prefix web run lint

dev-api:
	go run ./cmd/api

dev-web:
	npm --prefix web run dev

demo:
	docker compose up --build

catalog:
	.venv/bin/python ml/build_catalog.py

prepare-data:
	.venv/bin/python ml/prepare_goemotions.py

train:
	.venv/bin/python ml/train_transformer.py

export-model:
	.venv/bin/python ml/export_onnx.py
