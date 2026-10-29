# Deploying RAGOps for free: Render + Neon + Vercel (no credit card)

| Part | Service (free plan) | Notes |
|---|---|---|
| Backend API, workers, local models | **Render** web service | 512 MB RAM, 0.1 CPU, sleeps after 15 min idle |
| Database + vectors | **Neon** Postgres (pgvector) | 1 GB storage, 100 compute-hours/month, sleeps after 5 min idle |
| Frontend | **Vercel** | |
| Keep the backend awake | **cron-job.org** | pings `/ping` every 10 minutes |

## What runs where, and what to expect

The backend runs the same pipeline as on a big server, with every technique available.
The local models run on **ONNX Runtime instead of PyTorch**, which is what makes them fit in
512 MB:

* **Embeddings:** `all-MiniLM-L6-v2`, the same weights as before. Its vectors are identical to
  the PyTorch version (cosine 1.000).
* **Reranker:** `ms-marco-MiniLM-L-6-v2` (80 MB) instead of `bge-reranker-base` (1 GB, too big
  for 512 MB). On our labelled test set it put the right passage first as often as
  bge-reranker-base did (100% top-1), and the confidence gate is calibrated for it.

**Measured** (full ingest + chat cycle, PyTorch not installed): peak memory **468 MB**, and about
**0.35–0.45 CPU-seconds** per question. On Render's 0.1 CPU that is roughly **3–5 s** before the
answer starts streaming. Repeated questions come from the cache in well under a second.
Uploading a 50-page PDF takes about a minute in the background.

---

## Step 0: Snapshot your Neon database (1 minute, recommended)

The new version upgrades the database schema automatically. The change only adds columns and
tables, so nothing is deleted, but take a free snapshot first:

Neon console → your project → **Branches** → **Create branch** → name it `before-v2`, parent
`main`, *current point in time* → **Create**. That branch is a full copy you can return to.

## Step 1: Push the new code to GitHub

```bash
git checkout -b v2
git add -A
git commit -m "RAGOps v2"
git push -u origin v2
```

Then merge `v2` into `main` on GitHub when you're ready (Render deploys `main`).
**Do steps 2 and 3 before merging**, so the first deploy already uses the new settings.

## Step 2: Point Render at the new build and start commands

Render dashboard → your backend web service → **Settings**:

| Setting | Value |
|---|---|
| **Root Directory** | `backend` |
| **Build Command** | `pip install -r requirements.txt && python download_models.py` |
| **Start Command** | `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1 --proxy-headers --forwarded-allow-ips "*" --timeout-keep-alive 75` |
| **Health Check Path** | `/ping` (**important:** `/health` queries the database and would keep Neon awake 24/7) |
| **Instance Type** | Free |

(Starting fresh instead? Dashboard → **New → Blueprint** → pick the repo. The included
`render.yaml` creates the service with all of these settings.)

## Step 3: Environment variables on Render

Service → **Environment** → add or update:

| Key | Value |
|---|---|
| `PYTHON_VERSION` | `3.11.9` |
| `ENVIRONMENT` | `production` |
| `DATABASE_URL` | your Neon connection string (the **pooled** one, with `-pooler` in the host) |
| `SECRET_KEY` | a long random string (see below); keep it forever once set |
| `ADMIN_EMAIL` | your email |
| `ADMIN_PASSWORD` | a strong password (8+ characters) |
| `GROQ_API_KEY` / `GEMINI_API_KEY` / `OPENROUTER_API_KEY` | your keys (used only by your own workspace and the demo) |
| `DEMO_MODE` | `1` to show the one-click demo buttons on the login page |
| `MODEL_BACKEND` | `onnx` |
| `MODEL_CACHE_DIR` | `./.models` |
| `HF_HUB_OFFLINE` | `1` |
| `MALLOC_ARENA_MAX` | `2` |
| `WORKER_THREADS` | `1` |
| `CORS_ORIGINS` | only if your frontend uses a custom domain, e.g. `https://www.yoursite.com` |

To generate a `SECRET_KEY`, run this on your PC:
`python -c "import secrets; print(secrets.token_urlsafe(48))"`.
If you already had `SECRET_KEY` or `JWT_SECRET` set, keep the existing value.

Delete any old variables that set a different start command or `PRELOAD_MODELS=false`.

## Step 4: Deploy and watch the first start

Merge `v2` into `main` (or click **Manual Deploy → Deploy latest commit**). In **Logs**, expect:

1. The build installs packages and downloads about 175 MB of models (first build: several minutes).
2. `Schema upgraded, added columns: ...` and `Applied data migrations: ...`.
3. Your existing documents are queued for re-embedding into pgvector. They show as
   *processing* in the Knowledge Base tab for a while, then *complete*.
4. `Created admin account <your email>` (only if that account didn't exist yet).

Check it: open `https://<your-service>.onrender.com/ping` → `{"status":"ok"}`, then
`/health` → `{"status":"ok","database":true}`.

If you see `SECURITY: demo account admin@ragops.com still uses its public default password`,
handle it in step 7.

## Step 5: Keep the backend awake (free, no card)

A sleeping free Render service needs about a minute to wake up. To avoid that:

1. Sign up at <https://cron-job.org> (free).
2. **Create cronjob** → URL `https://<your-service>.onrender.com/ping` → every **10 minutes** → Save.

This uses about 744 of Render's 750 free instance-hours per month, so keep only **this one** free
Render service running. It pings `/ping`, which never touches the database, so Neon still sleeps
between questions and stays within its 100 compute-hours.

## Step 6: Point the frontend at the backend

Vercel → your project → **Settings → Environment Variables** →
`NEXT_PUBLIC_API_URL` = `https://<your-service>.onrender.com` (no trailing slash) →
**Deployments → ⋯ → Redeploy**. The redeploy is needed because this value is baked in at build time.

## Step 7: First login and clean-up

1. Log in with `ADMIN_EMAIL` / `ADMIN_PASSWORD`. Everything that existed before workspaces
   is now in **Main workspace**, which runs on your Render keys.
2. **Workspace & team:** delete `admin@ragops.com` and `client@ragops.com` if they're listed.
   Logins with their old public passwords are already refused. The demo buttons use separate
   `demo-admin@ragops.demo` / `demo-client@ragops.demo` accounts that have no password at all.
3. **Admin → RAG Configuration:** review each project's models and techniques.
4. Ask a question in **Chat**. The answer streams with numbered sources, and
   **Executions** shows the real timing of each step.

## Staying within the free limits

* **Neon compute (100 CU-hours/month):** each burst of activity keeps the database awake for 5
  minutes. A small team asking questions through the day fits comfortably. If the Neon console
  lets you set the compute size, keep it at the minimum (0.25 CU). Usage: Neon → **Billing**.
* **Neon storage (1 GB):** about 300,000 chunks. Deleting documents frees space.
* **Render memory:** if logs ever show *Out of memory*, set `RERANKER_MODEL=none`. That saves
  about 80 MB; relevance then comes from embeddings only, which is slightly less precise.
* **LLM free tiers:** each project has a fallback chain (Admin → RAG Configuration), so a
  rate-limited model automatically hands over to the next one.

## Troubleshooting

| Problem | Fix |
|---|---|
| Build fails on `download_models.py` | Re-deploy (Hugging Face download hiccup). Check `MODEL_CACHE_DIR=./.models`. |
| `Out of memory (used over 512Mi)` | `RERANKER_MODEL=none`, and make sure `WORKER_THREADS=1` and `MODEL_BACKEND=onnx`. |
| First answer after a long pause is slow | Normal if the cron ping isn't set up; also Neon wakes in about 1 s. |
| Browser shows a CORS error | Custom frontend domain missing from `CORS_ORIGINS`, or `NEXT_PUBLIC_API_URL` wrong or missing the redeploy. |
| Neon says compute quota reached | Check the Render health check path is `/ping`, not `/health`, and that nothing else polls the database. |
| Logins stop working after a deploy | `SECRET_KEY` changed. Set it back to the previous value. |

Moving to a bigger machine later (e.g. a VM): see `DEPLOY_ORACLE.md` and set `MODEL_BACKEND=torch`
to use `bge-reranker-base`.
