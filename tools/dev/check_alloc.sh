#!/bin/sh
# check_alloc.sh: run an athanor command and fail if any heap allocation
# comes from somewhere other than the arena allocator (AGENTS.md: memory
# comes from src/core arenas; step 1 acceptance criterion 4).
#
#   tools/dev/check_alloc.sh build/athanor inspect models/Qwen3.5-0.8B
#
# Development only: needs a C compiler and addr2line (binutils), and a
# release build (sanitizer builds cannot be combined with LD_PRELOAD).
set -eu

bin=$1
here=$(dirname "$0")
shim=$(dirname "$bin")/alloc_trace.so
cc -O1 -shared -fPIC -o "$shim" "$here/alloc_trace.c" -ldl

trace=$(LD_PRELOAD="$shim" "$@" 2>&1 >/dev/null | grep '^\[alloc\]' || true)
total=0
bad=0
# shellcheck disable=SC2034
while read -r tag fn bytes obj off; do
    [ -n "$tag" ] || continue
    total=$((total + 1))
    where=$(addr2line -f -e "$obj" "$off" 2>/dev/null | head -n 1)
    case "$where" in
        ath_arena_alloc) ;;
        *) bad=$((bad + 1)); echo "check_alloc: $fn of $bytes bytes from $where ($obj $off)" >&2 ;;
    esac
done <<EOF
$trace
EOF

if [ "$bad" -ne 0 ]; then
    echo "check_alloc: FAIL, $bad of $total allocations outside arenas" >&2
    exit 1
fi
echo "check_alloc: ok, $total allocations, all from ath_arena_alloc"
