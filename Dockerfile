FROM golang:1.26-alpine AS build

WORKDIR /src
COPY go.mod go.sum* ./
RUN go mod download
COPY cmd ./cmd
COPY internal ./internal
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/spotisense-api ./cmd/api

FROM alpine:3.21
RUN apk add --no-cache ca-certificates && addgroup -S app && adduser -S app -G app
WORKDIR /app
COPY --from=build /out/spotisense-api /usr/local/bin/spotisense-api
COPY data/tracks.csv /app/data/tracks.csv
USER app
EXPOSE 8080
ENTRYPOINT ["spotisense-api"]
