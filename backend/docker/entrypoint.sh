#!/bin/sh
# Usage: entrypoint.sh api | worker | migrate | <any command>
set -e

case "$1" in
  api)
    echo "Applying database migrations..."
    alembic upgrade head
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips='*'
    ;;
  worker)
    exec python -m app.workers.ingestion
    ;;
  migrate)
    exec alembic upgrade head
    ;;
  *)
    exec "$@"
    ;;
esac
