# Deploying RAGOps on Oracle Cloud (Always Free), step by step

> **Requires a credit/debit card** for Oracle's identity check. Without a card, use
> [DEPLOY_RENDER.md](./DEPLOY_RENDER.md) (Render + Neon + Vercel, no card).
> On a VM with 12 GB+ RAM you can set `MODEL_BACKEND=torch` (and the build arg of the same
> name) to use `bge-reranker-base`.

This puts the **whole backend** (API, background workers, embedding model, reranker,
PostgreSQL + pgvector) on one free Oracle Cloud ARM virtual machine, with free HTTPS.
The frontend stays on Vercel (or Cloudflare Pages). Your current Render + Neon setup keeps
running untouched until the very last step, so there is no downtime and nothing to lose.

**Time needed:** about 1–2 hours the first time (most of it waiting for builds).
**You need:** a credit/debit card for Oracle's identity check (Always Free resources are not
charged), your GitHub repository with the latest code pushed, and your API keys.

> Oracle's free allowance and policies change occasionally. Wherever this guide says
> "Always Free-eligible", trust the label in the Oracle console over this document.

---

## 1. Create the Oracle Cloud account

1. Go to <https://www.oracle.com/cloud/free/> and click **Start for free**.
2. Fill in your details. For **Home Region**, pick one close to you (for India: *India West
   (Mumbai)* or *India South (Hyderabad)*). **It cannot be changed later**, and free ARM
   machines are only created in your home region.
3. Verify your card. You may see a small temporary authorisation that is reversed.
4. Wait for the "account ready" email (minutes to an hour), then sign in at
   <https://cloud.oracle.com>.

## 2. Create the virtual machine

1. Menu (☰) → **Compute** → **Instances** → **Create instance**.
2. **Name:** `ragops`.
3. **Image and shape** → **Edit**:
   * **Change image** → **Canonical Ubuntu** → **24.04** (the normal one, not "Minimal").
     Make sure the image is the **aarch64** build.
   * **Change shape** → **Ampere** → **VM.Standard.A1.Flex**. Set **OCPUs = 2** and
     **Memory = 12 GB** (or more if the console still shows a larger Always Free allowance,
     up to 4 OCPU / 24 GB). Confirm the **Always Free-eligible** tag is shown.
4. **Networking:** keep "Create new virtual cloud network" and "Create new public subnet",
   and make sure **Assign a public IPv4 address** is on.
5. **Add SSH keys:** choose **Generate a key pair for me** and click **Save private key**.
   Keep this file safe; it is the only way into the machine.
6. **Boot volume:** tick **Specify a custom boot volume size** → **100 GB**
   (Always Free includes 200 GB of block storage in total).
7. Click **Create**. After 1–2 minutes the state turns **Running**. Copy the
   **Public IP address** shown on the instance page.

**"Out of capacity" error?** Free ARM machines are popular. Try another *Availability
domain* in the same form, try 1 OCPU / 6 GB and scale up later, or try again at a quieter
time (early morning). It always works eventually.

## 3. Connect from your Windows PC

Open **PowerShell** and run (replace the file name and IP):

```powershell
mkdir $HOME\.ssh -Force
Move-Item "$HOME\Downloads\ssh-key-*.key" "$HOME\.ssh\ragops.key"
icacls "$HOME\.ssh\ragops.key" /inheritance:r /grant:r "$($env:USERNAME):(R)"
ssh -i "$HOME\.ssh\ragops.key" ubuntu@<YOUR_VM_IP>
```

Type `yes` when asked about the fingerprint. You are now on the server
(the prompt looks like `ubuntu@ragops:~$`). Everything from here runs **on the server**.

## 4. Prepare the server

```bash
# Updates and basic tools
sudo apt update && sudo apt -y upgrade
sudo apt -y install git unattended-upgrades

# 4 GB swap as a safety net for builds
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Docker
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
exit
```

Reconnect with the same `ssh ...` command (so the Docker group applies), then check:

```bash
docker run --rm hello-world
```

## 5. Get the code and configure it

```bash
git clone https://github.com/<your-username>/<your-repo>.git RAGOps
cd RAGOps
cp .env.example .env
openssl rand -base64 48      # copy the output: this is your SECRET_KEY
openssl rand -base64 24      # copy the output: this is your POSTGRES_PASSWORD
nano .env
```

(For a private repository, create a GitHub *fine-grained personal access token* with
read access to the repo and use it as the password when `git clone` asks.)

In `nano`, set at least these lines, then save with **Ctrl+O, Enter** and exit with **Ctrl+X**:

```
SECRET_KEY=<the first random value>
ADMIN_EMAIL=<your email>
ADMIN_PASSWORD=<a strong password, 8+ characters>
POSTGRES_PASSWORD=<the second random value>
GROQ_API_KEY=<your key>
GEMINI_API_KEY=<your key>
OPENROUTER_API_KEY=<your key, optional>
ENVIRONMENT=production
```

Leave `DATABASE_URL` commented out: the VM runs its own PostgreSQL + pgvector, which is
faster than Neon's free tier and has no storage cap beyond your disk.

## 6. Bring your existing data over from Neon (optional but recommended)

This **copies** the data; Neon and Render keep working. In the Neon console, open your
project → **Connect**, turn **off** "Connection pooling", and copy that *direct* connection
string (its host does **not** contain `-pooler`).

```bash
cd ~/RAGOps
docker compose up -d db                     # start only the database
docker run --rm postgres:17-alpine pg_dump "<NEON_DIRECT_URL>" \
  --no-owner --no-acl -Fc > ~/neon.dump
ls -lh ~/neon.dump                          # should be more than a few KB
docker compose exec -T db sh -c 'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --no-owner --no-acl --clean --if-exists' < ~/neon.dump
```

A few "does not exist, skipping" messages from `--clean` are normal. On the first start,
the backend upgrades the schema automatically and **re-embeds every existing document into
pgvector** in the background (the old FAISS index files were never stored in the database).

## 7. Build and start the backend

```bash
cd ~/RAGOps
docker compose up -d --build
```

The first build takes 10–20 minutes on ARM (it installs CPU-only PyTorch and downloads the
embedding and reranker models into the image). Then check:

```bash
docker compose ps                      # backend should become "healthy"
curl http://localhost:8000/health      # {"status":"ok","database":true}
docker compose logs -f backend         # Ctrl+C to stop following the logs
```

Want better PDF parsing (tables, headings, scanned pages)? Set `INSTALL_DOCLING=true` in
`.env` and run `docker compose up -d --build backend` (the image becomes larger).

## 8. Give the backend a public HTTPS address

Your frontend (served over HTTPS) can only call an HTTPS backend. Pick **one** option.

### Option A: free DuckDNS name + automatic certificate (no domain needed)

1. Go to <https://www.duckdns.org>, sign in (GitHub/Google), create a subdomain such as
   `ragops-amritanshu`, put your **VM public IP** in the *current ip* box, and click
   **update ip**.
2. Open ports 80 and 443 in Oracle's firewall: Oracle console → your instance →
   **Subnet** link → **Security Lists** → *Default Security List* →
   **Add Ingress Rules**: Source CIDR `0.0.0.0/0`, IP protocol TCP,
   Destination port range `80,443` → **Add**.
3. Open them on the server too (Oracle's Ubuntu images block them by default):

   ```bash
   sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
   sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
   sudo netfilter-persistent save
   ```

4. Add your name to `.env` (`nano .env`):

   ```
   API_DOMAIN=ragops-amritanshu.duckdns.org
   ```

5. Start the HTTPS proxy:

   ```bash
   docker compose --profile https up -d
   docker compose logs caddy | tail -20     # look for "certificate obtained successfully"
   ```

6. From your PC, open `https://ragops-amritanshu.duckdns.org/health`; you should see
   `{"status":"ok","database":true}`.

### Option B: Cloudflare Tunnel (only if you own a domain managed by Cloudflare)

No ports need opening. In the Cloudflare dashboard: **Zero Trust → Networks → Tunnels →
Create a tunnel → Cloudflared**, name it `ragops`, copy the **token**. Under **Public
Hostname** add `api.yourdomain.com` → service `http://backend:8000`. Then on the server:

```bash
echo "CLOUDFLARE_TUNNEL_TOKEN=<token>" >> .env
docker compose --profile tunnel up -d
```

## 9. Point the frontend at the new backend

**Vercel:** Project → **Settings → Environment Variables** → set `NEXT_PUBLIC_API_URL` to
`https://ragops-amritanshu.duckdns.org` (your Option A or B address, no trailing slash) for
*Production* → **Deployments** → ⋯ on the latest → **Redeploy**. A redeploy is required:
this value is baked in at build time.

If your frontend uses a custom domain (not `*.vercel.app` / `*.pages.dev`), add it on the
server and restart the backend:

```bash
echo "CORS_ORIGINS=https://www.your-frontend-domain.com" >> ~/RAGOps/.env
docker compose up -d backend
```

## 10. First login and checks

1. Open your site and log in with `ADMIN_EMAIL` / `ADMIN_PASSWORD` from step 5.
2. **Admin → Access & Integrations → Accounts:** delete `admin@ragops.com` and
   `client@ragops.com` if they exist. Older versions created them with public default
   passwords (the backend log warns about them on startup).
3. **Admin → Knowledge Base:** documents brought over from Neon show as processing for a
   few minutes while they are re-embedded, then *complete*.
4. Ask a question in **Chat**: the answer should stream in with numbered sources.
5. **Executions** (`/executions`) shows the real trace and timing of every question.

## 11. Backups, updates and monitoring

**Nightly database backup** (keeps 7 days in `~/backups`):

```bash
chmod +x ~/RAGOps/deploy/backup.sh
~/RAGOps/deploy/backup.sh                    # test once
crontab -e                                   # choose nano, then add this line:
30 2 * * * /home/ubuntu/RAGOps/deploy/backup.sh >> /home/ubuntu/backup.log 2>&1
```

Copy a backup to your PC now and then (run this **on your PC**):

```powershell
scp -i "$HOME\.ssh\ragops.key" "ubuntu@<YOUR_VM_IP>:~/backups/*.dump" .
```

**Deploying new code:**

```bash
cd ~/RAGOps && git pull && docker compose up -d --build backend
```

**Uptime monitoring:** create a free monitor at <https://uptimerobot.com> for
`https://<your-domain>/health` (every 5 minutes). It emails you if the backend goes down.

**Useful commands:** `docker compose ps`, `docker compose logs -f backend`,
`docker compose restart backend`, `df -h` (disk), `free -h` (memory).

## 12. Keeping the free VM

Oracle can reclaim Always Free instances that stay almost completely idle for a week
(very low CPU, network and memory use). A RAGOps instance that people actually use, plus
nightly backups and the uptime monitor, is not idle in that sense. Two further options:

* Upgrade the account to **Pay As You Go** (Billing → Upgrade). Always Free resources stay
  free, and idle reclamation is reported not to apply to paid accounts. If you do this,
  create a **budget alert** of a few dollars (Billing → Budgets) so any accidental paid
  resource is noticed immediately.
* Don't run fake "keep-busy" load scripts: Oracle treats them as abuse.

Keep your backups (step 11) regardless: if the VM is ever lost, a new VM plus
`pg_restore` of the latest dump restores everything.

## 13. Switch off Render

After a few days of everything working on Oracle:

1. Render dashboard → your backend service → **Settings** → **Delete Web Service**.
2. Neon: keep the project as an extra off-site copy, or delete it from the Neon console.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Out of capacity` when creating the VM | Another availability domain, fewer OCPUs, or retry later. |
| `ssh: Permission denied (publickey)` | Use `ubuntu@`, the right IP, and the `.key` file you saved; re-run the `icacls` line. |
| HTTPS address doesn't load | Check the DuckDNS IP matches the VM, the security-list rule (step 8A.2), the `iptables` lines (8A.3), and `docker compose logs caddy`. |
| Browser console shows a CORS error | Frontend domain missing from `CORS_ORIGINS`, or `NEXT_PUBLIC_API_URL` has a trailing slash / wrong scheme; redeploy the frontend after changing it. |
| Backend not healthy | `docker compose logs backend`; most often a missing or invalid value in `.env`. |
| Answers say "insufficient confidence" too often | Admin → RAG Configuration → *Confidence gate*: lower it slightly (0.40 is the calibrated default). |
| Out of disk | `docker system prune -af` removes old images; check `~/backups`. |
