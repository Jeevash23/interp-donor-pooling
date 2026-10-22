#!/bin/sh
# Billing watchdog for long Study 5 stages: every 15 minutes, read Modal's billing report; if every Study 5 app
# (names starting with $STUDY5_APP) together has spent $10 or more, stop the running ones (PROTOCOL.md,
# "Stopping"). Logs to results/billing-watch.jsonl. Exits when no Study 5 app is running. Run from this folder.
modal="${MODAL_CLI:-modal}"
export STUDY5_APP="${STUDY5_APP:-donor-pooling-larger-model}"
while true; do
    spend=$("$modal" billing report --for "this month" --json 2>/dev/null | python3 -c '
import json, os, sys
try:
    rows = json.load(sys.stdin)
    print(round(sum(float(r["cost"]) for r in rows if r["description"].startswith(os.environ["STUDY5_APP"])), 4))
except Exception:
    print("unknown")')
    running=$("$modal" app list --json 2>/dev/null | python3 -c '
import json, os, sys
try:
    apps = json.load(sys.stdin)
    print(" ".join(a["app_id"] for a in apps if a["description"] == os.environ["STUDY5_APP"] and a["state"] != "stopped"))
except Exception:
    print("unknown")')
    echo "{\"utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"study5_usd\": \"$spend\", \"running\": \"$running\"}" >> results/billing-watch.jsonl
    if [ "$spend" != "unknown" ] && python3 -c "import sys; sys.exit(0 if float('$spend') >= 10 else 1)"; then
        for app in $running; do "$modal" app stop -y "$app"; done
        echo "{\"utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"stopped\": \"$running\", \"reason\": \"spend reached \$10\"}" >> results/billing-watch.jsonl
        exit 0
    fi
    [ -z "$running" ] && exit 0
    sleep 900
done
