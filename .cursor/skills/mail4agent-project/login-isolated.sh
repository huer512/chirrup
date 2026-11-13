#!/usr/bin/env bash
# Login Cursor Agent for mail4agent automatic reply (isolated credential store only).
set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"

CONN="${HOME}/.local/share/mail4agent/project-connections/eea28946-135a-4fe5-b67e-bf1480442716"
export CURSOR_CONFIG_DIR="${CONN}/private/cursor-config"
export CURSOR_DATA_DIR="${CONN}/private/cursor-data"
export XDG_CONFIG_HOME="${CONN}/private/cursor-xdg/config"
export XDG_DATA_HOME="${CONN}/private/cursor-xdg/data"
export XDG_CACHE_HOME="${CONN}/private/cursor-xdg/cache"
export AGENT_CLI_CREDENTIAL_STORE="file"

echo "Connection: ${CONN}"
echo "Before login:"
agent status || true
echo ""
echo "Starting agent login (open the printed link in your browser and approve)..."
agent login
echo ""
echo "After login:"
agent status
