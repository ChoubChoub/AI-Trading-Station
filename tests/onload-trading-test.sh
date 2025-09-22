#!/usr/bin/env bash
set -e

echo "== onload-trading /bin/echo OK =="
onload-trading /bin/echo OK

echo
echo "== onload-trading /usr/bin/env | grep -E '^(EF_|LD_PRELOAD)=' =="
onload-trading /usr/bin/env | grep -E '^(EF_|LD_PRELOAD)='

