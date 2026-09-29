#!/bin/bash
# Nightly backup to another machine over rsync+SSH. The server is stopped for the copy so the
# database file and its WAL are consistent; downtime is a few seconds. Relays
# retry when /input is unreachable, so messages sent meanwhile arrive late, not never.
set -euo pipefail
. /etc/agent/backup.env   # BACKUP_TARGET=user@host, optionally BACKUP_SSH_PORT=22
KEEP_DAYS=30
SSH=(ssh -p "${BACKUP_SSH_PORT:-22}" -i /etc/agent/backup_key -o BatchMode=yes)
stamp=$(date -u +%Y-%m-%dT%H%MZ)
work=$(mktemp -d)
trap 'rm -rf "$work"; systemctl start agent' EXIT

systemctl stop agent
mkdir "$work/db"
cp -a /var/lib/agent/. "$work/db/"
systemctl start agent

tar -C "$work" -czf "$work/agent-$stamp.tar.gz" db
rsync -e "${SSH[*]}" "$work/agent-$stamp.tar.gz" "$BACKUP_TARGET:backups/"

# Prune old backups.
cutoff=$(date -u -d "-$KEEP_DAYS days" +%Y-%m-%d)
"${SSH[@]}" "$BACKUP_TARGET" ls backups | while read -r f; do
  if [[ $f =~ ^agent-([0-9]{4}-[0-9]{2}-[0-9]{2}) ]] && [[ ${BASH_REMATCH[1]} < $cutoff ]]; then
    "${SSH[@]}" "$BACKUP_TARGET" rm "backups/$f"
  fi
done
# Tell the server when the last backup succeeded; GET /health reports its age.
date -u +%Y-%m-%dT%H:%M:%SZ > /var/lib/agent/last-backup
chown agent:agent /var/lib/agent/last-backup
echo "backed up agent-$stamp.tar.gz"
