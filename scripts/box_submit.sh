#!/usr/bin/env bash
# Runs HERE. Usage: [BOX_ENV="K=V ..."] box_submit.sh <log-name> <gpu|cpu> <cargo test filter...>
# BOX_ENV is passed to the job's environment, e.g. BOX_ENV="TIG_TEST_EXTRA=--ignored".
# BOX_NO_WAIT=1 prints the job id and exits 0 immediately after submitting,
# skipping `gpuq wait` and `gpuq show`. Default behaviour (wait) is unchanged.
# Refuses to submit a commit the box cannot fetch.
set -euo pipefail
name=$1
lane=$2
shift 2
sha=$(git rev-parse HEAD)
br=$(git rev-parse --abbrev-ref HEAD)
git fetch -q origin "$br"
if [ "$(git rev-parse "origin/$br")" != "$sha" ]; then
    echo "HEAD $sha is not what origin/$br points at; push first" >&2
    exit 2
fi
remote=(gpuq submit --project tig-monorepo --commit "$sha" --branch "$br" --lane "$lane" --timeout-s "${BOX_TIMEOUT_S:-5400}" -- env)
# BOX_ENV is a space-separated list of K=V words by contract; split it on purpose.
read -r -a box_env <<< "${BOX_ENV:-}"
remote+=("${box_env[@]+"${box_env[@]}"}" bash scripts/box_test.sh "$name" "$@")
cmd=$(printf '%q ' "${remote[@]}")
id=$(ssh -o BatchMode=yes tig-gpu "$cmd" 2>/dev/null | tail -1)
echo "job: $id"
if [ -z "$id" ]; then
    echo "gpuq submit produced no job id" >&2
    exit 3
fi
if [ "${BOX_NO_WAIT:-0}" = "1" ]; then
    exit 0
fi
rc=0
ssh -o BatchMode=yes tig-gpu "gpuq wait $id" 2>/dev/null || rc=$?
ssh -o BatchMode=yes tig-gpu "gpuq show $id" 2>/dev/null
exit $rc
