#!/bin/sh
# Run one Study 5 stage on Modal as a standalone process. The local connection sometimes fails to connect or to
# resolve names; when that happens before the stage has started (no new line in results/stage-log.jsonl, so no
# GPU work was requested), the idle app is stopped and the stage is retried. Usage, from this folder:
#   systemd-run --user --unit=study5-<stage> --working-directory="$PWD" ./run_stage.sh <stage>
stage="$1"
log="logs/$stage.log"
modal="${MODAL_CLI:-modal}"
mkdir -p logs
lines() { [ -f results/stage-log.jsonl ] && wc -l < results/stage-log.jsonl || echo 0; }
for attempt in 1 2 3 4 5 6; do
    before=$(lines)
    PYTHONDONTWRITEBYTECODE=1 "$modal" run --detach "modal_app.py::$stage" > "$log" 2>&1
    status=$?
    echo "exit $status (attempt $attempt)" >> "$log"
    if [ $status -eq 0 ] || [ "$(lines)" != "$before" ] || \
       ! grep -q -e "Could not connect to the Modal server" -e "Temporary failure in name resolution" "$log"; then
        echo "exit $status" >> "$log"
        exit $status
    fi
    app=$(grep -o "ap-[A-Za-z0-9]*" "$log" | head -1)
    [ -n "$app" ] && "$modal" app stop -y "$app" >> "$log" 2>&1
    mv "$log" "logs/$stage-connect-failure-$(date -u +%H%M%S).log"
    sleep 30
done
echo "exit 1" >> "$log"
exit 1
