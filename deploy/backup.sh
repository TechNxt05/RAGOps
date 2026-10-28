#!/usr/bin/env bash
# Nightly Postgres backup, keeping the last 7 days.
# Install:  crontab -e   then add:   30 2 * * * /home/ubuntu/RAGOps/deploy/backup.sh >> /home/ubuntu/backup.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p "$HOME/backups"
STAMP=$(date +%Y%m%d-%H%M)
docker compose exec -T db sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > "$HOME/backups/ragops-$STAMP.dump"
find "$HOME/backups" -name 'ragops-*.dump' -mtime +7 -delete
echo "$(date) backup ok: ragops-$STAMP.dump ($(du -h "$HOME/backups/ragops-$STAMP.dump" | cut -f1))"
