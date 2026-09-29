# Sourced by the deploy scripts: loads deploy/config and sets HOST.
config="$(dirname "${BASH_SOURCE[0]}")/config"
[ -f "$config" ] || { echo "missing $config; copy deploy/config.example and fill it in" >&2; exit 1; }
. "$config"
HOST=$AGENT_HOST
