# Free-tier deployment runbook

This runbook deploys the React client to Vercel, the Go API to Google Cloud Run, PostgreSQL to Neon, and Redis to Upstash. It assumes the repository has already been pushed to GitHub.

Free allowances are sufficient for typical demo traffic, but they are not a guarantee of a zero bill. Cloud Run requires a Google Cloud project with billing enabled; create a small budget alert before deploying and keep the service minimum at zero.

## 1. Create the managed data services

Create a free Neon project. In its **Connect** dialog, enable connection pooling and copy the URL containing `-pooler`; Neon recommends the pooled URL for serverless applications.

Create a free Upstash Redis database. Copy its TLS Redis URL from the database details. It should begin with `rediss://`, not the REST URL.

Do not put either credential in Git. At the repository root, create `.cloudrun.env.yaml`:

```yaml
APP_ENV: "production"
TRACKS_PATH: "/app/data/tracks.csv"
DATABASE_URL: "postgresql://USER:PASSWORD@ENDPOINT-pooler.REGION.aws.neon.tech/neondb?sslmode=require"
REDIS_URL: "rediss://default:PASSWORD@ENDPOINT.upstash.io:PORT"
CORS_ALLOWED_ORIGINS: "http://localhost:5173"
SPOTIFY_CLIENT_ID: ""
SPOTIFY_CLIENT_SECRET: ""
```

The file is already ignored by Git.

## 2. Deploy the Go API to Cloud Run

Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install), then run:

```bash
gcloud auth login
gcloud projects create YOUR_UNIQUE_PROJECT_ID --name="SpotiSense"
gcloud config set project YOUR_UNIQUE_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy spotisense-api \
  --source . \
  --region northamerica-northeast1 \
  --allow-unauthenticated \
  --env-vars-file .cloudrun.env.yaml \
  --cpu 1 \
  --memory 512Mi \
  --min 0 \
  --max 3
```

If the project already exists, skip `gcloud projects create`. Google may ask you to link a billing account before deployment.

Capture and test the generated API URL:

```bash
API_URL="$(gcloud run services describe spotisense-api --region northamerica-northeast1 --format='value(status.url)')"
curl "$API_URL/healthz"
```

The health response should report `postgres`, `redis`, and `750` catalog tracks.

## 3. Verify the browser build

The tested quantized model is already part of the repository. Verify it and build the client:

```bash
test -f web/public/models/spotisense-emotion/onnx/model_quantized.onnx
npm --prefix web ci
npm --prefix web run build
```

Retraining is optional and is documented in the main README. It is not required for deployment.

## 4. Deploy the React client to Vercel

In the Vercel dashboard:

1. Select **Add New → Project** and import the GitHub repository.
2. Set **Root Directory** to `web`; Vercel will detect Vite.
3. Add these **Production** environment variables:

```text
VITE_API_URL=<your Cloud Run URL>
VITE_EMOTION_MODEL=/models/spotisense-emotion
```

4. Select **Deploy** and copy the resulting `https://YOUR-PROJECT.vercel.app` URL.

Once connected, pushes to the production branch trigger future deployments automatically. Preview deployments use different origins; either test the production deployment only or add a specific preview origin to `CORS_ALLOWED_ORIGINS` before using it.

## 5. Allow the Vercel origin

Edit `.cloudrun.env.yaml` and replace the temporary `CORS_ALLOWED_ORIGINS` value:

```yaml
CORS_ALLOWED_ORIGINS: "https://YOUR-PROJECT.vercel.app"
```

Redeploy the API configuration:

```bash
gcloud run deploy spotisense-api \
  --source . \
  --region northamerica-northeast1 \
  --allow-unauthenticated \
  --env-vars-file .cloudrun.env.yaml \
  --cpu 1 \
  --memory 512Mi \
  --min 0 \
  --max 3
```

If you use a custom domain, include both origins separated by a comma.

## 6. Verify the production path

Open the Vercel URL in a private browser window and verify:

1. A profile can be created.
2. Emotion analysis says **Fine-tuned browser model**, not **Development fallback**.
3. A recommendation returns six or more tracks.
4. A playlist can be created and a track saved.
5. Reloading preserves the profile, playlist, and history.
6. Cloud Run `/healthz` reports PostgreSQL and Redis rather than memory adapters.

Use the browser Network panel to confirm that the mood sentence never leaves the browser. The recommendation request contains only a SHA-256 context key for diversification, and persisted history contains only emotion, confidence, and tracks.

## 7. Cost controls

- Keep Cloud Run `--min 0` and cap `--max 3`.
- Configure a Google Cloud budget alert; a budget alert notifies you but does not automatically stop spending.
- Do not use artificial keep-alive requests—the service should scale to zero.
- Review Neon, Upstash, Vercel, and Cloud Run usage after the first complete test.
- Delete unused preview deployments and rotate credentials if they ever appear in logs or Git history.

Current provider references: [Cloud Run source deployment](https://cloud.google.com/run/docs/deploying-source-code), [Cloud Run configuration](https://cloud.google.com/run/docs/configuring), [Vercel monorepos](https://vercel.com/docs/monorepos), [Neon pooled connections](https://neon.com/docs/connect/connection-pooling), and [Upstash TLS](https://upstash.com/docs/redis/features/security).
