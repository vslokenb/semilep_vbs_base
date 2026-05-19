#!/usr/bin/env bash
# Run pocket-coffea build-datasets for every discovery JSON (excluding _replicas files).

DISCOVERY_DIR="${1:-datasets/discovery}"

if [[ ! -d "$DISCOVERY_DIR" ]]; then
    echo "ERROR: directory not found: $DISCOVERY_DIR"
    exit 1
fi

shopt -s nullglob
files=("$DISCOVERY_DIR"/*.json)

if [[ ${#files[@]} -eq 0 ]]; then
    echo "No JSON files found in $DISCOVERY_DIR"
    exit 1
fi

ok=0; fail=0
for f in "${files[@]}"; do
    [[ "$f" == *_replicas.json ]] && continue
    echo "==> $f"
    if pocket-coffea build-datasets --cfg "$f" --overwrite; then
        (( ok++ ))
    else
        echo "    FAILED: $f"
        (( fail++ ))
    fi
done

echo ""
echo "Done: $ok succeeded, $fail failed."
[[ $fail -eq 0 ]]
