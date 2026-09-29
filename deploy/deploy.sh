#!/bin/bash
# Build on the VM and (re)install. Usage: deploy/deploy.sh
set -euo pipefail
. "$(dirname "$0")/lib.sh"
cd "$(dirname "$0")/.."
ssh "$HOST" mkdir -p /root/src/agent
rsync -az --delete --exclude target --exclude .git --exclude /.env --exclude /deploy/config ./ "$HOST:/root/src/agent/"
ssh "$HOST" bash -se -- "$AGENT_PRIVATE_IP" <<'REMOTE'
set -euo pipefail
PRIVATE_IP=$1
cd /root/src/agent
~/.cargo/bin/cargo build --release --locked
install -m 0755 target/release/agent /usr/local/bin/agent
install -m 0755 deploy/backup.sh /usr/local/bin/agent-backup
install -m 0644 deploy/agent-backup.service deploy/agent-backup.timer /etc/systemd/system/
# The private IP lives in the untracked deploy/config, not in the repo.
sed "s/@PRIVATE_IP@/$PRIVATE_IP/g" deploy/agent.service > /etc/systemd/system/agent.service
[ -f /etc/agent/env ] || { echo "missing /etc/agent/env; run deploy/secrets.sh first" >&2; exit 1; }
mountpoint -q /var/lib/agent || { echo "/var/lib/agent isn't mounted" >&2; exit 1; }
install -d -o agent -g agent -m 0700 /var/lib/agent/home /var/lib/agent/work
systemctl daemon-reload
systemctl enable --now agent agent-backup.timer
systemctl restart agent
systemctl --no-pager --lines=5 status agent
REMOTE
