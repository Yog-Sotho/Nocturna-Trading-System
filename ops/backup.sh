#!/bin/sh
# NOCTURNA Trading System - database backups (the compose `backup` service).
#
# Writes a pg_dump custom-format file to /backups BACKUP_FIRST_DELAY_SECONDS
# after start-up (default 10 minutes, so a first start dumps the migrated
# schema, not an empty database) and then every BACKUP_INTERVAL_SECONDS
# (default one day), and deletes dumps older than BACKUP_KEEP_DAYS (default 14). With BACKUP_HEARTBEAT_URL (a healthchecks.io
# ping URL) each run pings it, or its /fail on a failed dump, so a backup that
# silently stops running is noticed. Restoring: docs/RUNBOOK.md.
#
# Connection settings come from PGHOST, PGUSER, PGPASSWORD and PGDATABASE.
set -eu

interval="${BACKUP_INTERVAL_SECONDS:-86400}"
keep="${BACKUP_KEEP_DAYS:-14}"
heartbeat="${BACKUP_HEARTBEAT_URL:-}"

ping() {
    # Never print the URL: it is the check's secret
    if [ -n "$heartbeat" ] && ! wget -q -T 10 -O /dev/null "$1" 2>/dev/null; then
        echo "backup: heartbeat ping failed" >&2
    fi
}

sleep "${BACKUP_FIRST_DELAY_SECONDS:-600}"
while true; do
    file="/backups/nocturna-$(date -u +%Y%m%dT%H%M%SZ).dump"
    if pg_dump --format=custom --file="$file.partial"; then
        mv "$file.partial" "$file"
        echo "backup: wrote $file ($(du -h "$file" | cut -f1))"
        ping "$heartbeat"
    else
        rm -f "$file.partial"
        echo "backup: pg_dump failed" >&2
        ping "${heartbeat%/}/fail"
    fi
    find /backups -name 'nocturna-*.dump' -type f -mtime "+$keep" -exec rm -f {} +
    sleep "$interval"
done
