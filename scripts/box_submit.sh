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

# ssh's stderr goes to a file rather than /dev/null, and every failure path
# prints it. Discarding it made an unreachable host and a failed submit produce
# the same message, which is the one distinction worth having here. The job id is
# still read from stdout only, so a connect-timeout line cannot be mistaken for
# one. Exit 4 means ssh itself failed; exit 3 means ssh succeeded and no job id
# came back.
err=$(mktemp)
trap 'rm -f "$err"' EXIT

rc=0
id=$(ssh -o BatchMode=yes tig-gpu "$cmd" 2>"$err" | tail -1) || rc=$?
if [ "$rc" -ne 0 ]; then
    echo "ssh failed (exit $rc); the box may be unreachable. ssh said:" >&2
    cat "$err" >&2
    exit 4
fi
echo "job: $id"
if [ -z "$id" ]; then
    echo "gpuq submit produced no job id; ssh succeeded. ssh said:" >&2
    cat "$err" >&2
    exit 3
fi
if [ "${BOX_NO_WAIT:-0}" = "1" ]; then
    exit 0
fi
rc=0
ssh -o BatchMode=yes tig-gpu "gpuq wait $id" 2>"$err" || rc=$?
if [ "$rc" -ne 0 ]; then
    echo "gpuq wait exited $rc. ssh said:" >&2
    cat "$err" >&2
fi
ssh -o BatchMode=yes tig-gpu "gpuq show $id" 2>"$err" || cat "$err" >&2
exit $rc
