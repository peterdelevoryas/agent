#!/bin/bash
# Write the server's /etc/agent/env from secrets on this Mac, then restart the
# agent if it's installed. Nothing is printed. Sources:
#   ANTHROPIC_API_KEY       from ../.env
#   AGENT_INPUT_TOKEN       keychain agent-input-token (created on first run);
#                           relays send it as their bearer token
#   AGENT_MCP_TOKEN_<NAME>  keychain <name>-mcp-token / agent, per AGENT_MCP_SERVERS
# Usage: deploy/secrets.sh
set -euo pipefail
. "$(dirname "$0")/lib.sh"
dotenv="$(dirname "$0")/../.env"
api_key=$(sed -n 's/^\(export \)\{0,1\}ANTHROPIC_API_KEY=//p' "$dotenv" | tr -d "\"'")
[ -n "$api_key" ] || { echo "no ANTHROPIC_API_KEY in $dotenv" >&2; exit 1; }
if ! input_token=$(security find-generic-password -s agent-input-token -a agent -w 2>/dev/null); then
  input_token="ag_$(openssl rand -hex 32)"
  security add-generic-password -s agent-input-token -a agent -w "$input_token"
  echo "created a new input token (keychain agent-input-token / agent)" >&2
fi
env="ANTHROPIC_API_KEY=$api_key
AGENT_INPUT_TOKEN=$input_token
AGENT_MCP_SERVERS=$AGENT_MCP_SERVERS
AGENT_RELAYS=${AGENT_RELAYS:-}
"
IFS=',' read -ra servers <<< "$AGENT_MCP_SERVERS"
for entry in "${servers[@]}"; do
  name=${entry%%=*}
  token=$(security find-generic-password -s "$name-mcp-token" -a agent -w) \
    || { echo "no keychain token $name-mcp-token / agent" >&2; exit 1; }
  env+="AGENT_MCP_TOKEN_$(echo "$name" | tr '[:lower:]' '[:upper:]')=$token
"
done
printf '%s' "$env" | ssh "$HOST" 'install -m 0640 -o root -g agent /dev/stdin /etc/agent/env && if systemctl cat agent >/dev/null 2>&1; then systemctl reset-failed agent; systemctl restart agent; fi'
echo "installed /etc/agent/env on $HOST" >&2
