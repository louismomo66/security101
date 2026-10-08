#!/usr/bin/env bash
#
# Point the frontend at a remote Sentinel backend and start it.
#
#   ./connect.sh https://something.trycloudflare.com
#   ./connect.sh                    # reuses the last URL
#
# Why this exists: free Cloudflare quick tunnels mint a NEW hostname every
# time they start, and Next.js bakes NEXT_PUBLIC_API_URL into the bundle at
# build time. So a restarted tunnel leaves the browser calling an address
# that no longer resolves, and refreshing the page cannot fix it — the dead
# URL is compiled into .next. Every reconnect therefore needs the cache
# cleared, which is easy to forget and confusing when you do.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LAST="$HERE/.last-backend-url"

URL="${1:-}"
if [ -z "$URL" ]; then
  [ -f "$LAST" ] || { echo "usage: $0 <backend-url>"; exit 1; }
  URL="$(cat "$LAST")"
  echo "reusing last URL: $URL"
fi
URL="${URL%/}"

echo "▶ checking $URL …"
if ! curl -fsS --max-time 25 "$URL/api/weapons/status" >/dev/null 2>&1; then
  echo "✗ backend did not answer at $URL/api/weapons/status"
  echo "  Is the tunnel up on the GPU box? Quick tunnels change hostname on"
  echo "  every restart, so yesterday's URL will not work today."
  exit 1
fi
echo "✓ backend answered"
echo "$URL" > "$LAST"

# The stale URL lives in the build cache, not just in memory — clearing it is
# the whole point of this script.
rm -rf "$HERE/.next"

PORT="${PORT:-3000}"
if lsof -ti:"$PORT" >/dev/null 2>&1; then
  echo "▶ freeing port $PORT"
  lsof -ti:"$PORT" | xargs kill 2>/dev/null || true
  sleep 2
fi

echo "▶ starting frontend against $URL"
echo "  then open http://localhost:$PORT and hard-reload (Cmd+Shift+R)"
cd "$HERE"
NEXT_PUBLIC_API_URL="$URL" npm run dev
