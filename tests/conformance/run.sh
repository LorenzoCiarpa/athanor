#!/bin/sh
# run.sh <harness> <task|all> <outdir>: the conformance task set of step 13
# through a coding agent driving `athanor serve`.
#
#   harness   claude (Claude Code, Anthropic Messages), codex (Codex CLI,
#             OpenAI Responses) or opencode (OpenCode, OpenAI Chat Completions)
#   task      a folder of tasks/, or all
#   outdir    where transcripts, diffs and results go
#
# Each task runs in a fresh git repository under $TMPDIR (Claude Code refuses
# to edit files under ~/.claude) made from tasks/<task>/files, with the
# prompt of tasks/<task>/prompt.txt; then tasks/<task>/check.sh decides the
# outcome in that folder. The harness gets a configuration folder of its
# own under $TMPDIR: the user's ~/.claude, ~/.codex and ~/.config/opencode
# are never read or written. The commands the agent may run are limited to
# a list (claude, opencode) or by Codex's workspace-write sandbox.
#
# Environment: ATHANOR_URL (default http://127.0.0.1:8080), ATHANOR_MODEL
# (the name the harness sends; default qwen3.6-35b-a3b), ATHANOR_CTX (the
# server's context, for the harness's compaction; default 65536).
set -u
here=$(cd "$(dirname "$0")" && pwd)
[ $# -eq 3 ] || { echo "usage: $0 <claude|codex|opencode> <task|all> <outdir>" >&2; exit 2; }
harness=$1 which=$2
mkdir -p "$3" && out=$(cd "$3" && pwd)
url=${ATHANOR_URL:-http://127.0.0.1:8080}
model=${ATHANOR_MODEL:-qwen3.6-35b-a3b}
ctx=${ATHANOR_CTX:-65536}
home="${TMPDIR:-/tmp}/athanor-conformance-home-$harness"
mkdir -p "$home"

curl -sf "$url/health" >/dev/null || { echo "no athanor serve at $url" >&2; exit 1; }

case $harness in
claude)
    command -v claude >/dev/null || { echo "claude not found" >&2; exit 1; }
    version=$(claude --version 2>/dev/null)
    ;;
codex)
    command -v codex >/dev/null || { echo "codex not found" >&2; exit 1; }
    version=$(codex --version 2>/dev/null)
    mkdir -p "$home/codex"
    cat > "$home/codex/config.toml" <<EOF
model = "$model"
model_provider = "athanor"
model_context_window = $ctx
model_reasoning_effort = "medium"

[model_providers.athanor]
name = "athanor"
base_url = "$url/v1"
wire_api = "responses"
env_key = "ATHANOR_KEY"
stream_idle_timeout_ms = 1800000
request_max_retries = 1
stream_max_retries = 1
EOF
    ;;
opencode)
    command -v opencode >/dev/null || { echo "opencode not found" >&2; exit 1; }
    version="opencode $(opencode --version 2>/dev/null)"
    mkdir -p "$home/opencode/config/opencode"
    cat > "$home/opencode/config/opencode/opencode.json" <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "athanor": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "athanor",
      "options": {"baseURL": "$url/v1", "apiKey": "athanor-local", "timeout": false},
      "models": {
        "$model": {"name": "$model (athanor)", "reasoning": true, "tool_call": true,
                   "limit": {"context": $ctx, "output": 16384}}
      }
    }
  },
  "model": "athanor/$model",
  "small_model": "athanor/$model",
  "autoupdate": false,
  "share": "disabled",
  "permission": {
    "edit": "allow",
    "webfetch": "deny",
    "bash": {"*": "deny", "python3 *": "allow", "ls*": "allow", "cat *": "allow", "head *": "allow",
             "grep *": "allow", "find *": "allow", "wc *": "allow", "git diff*": "allow", "git status*": "allow"}
  }
}
EOF
    ;;
*)
    echo "unknown harness $harness" >&2; exit 2 ;;
esac

run_claude() {
    CLAUDE_CONFIG_DIR="$home/claude" ANTHROPIC_BASE_URL="$url" ANTHROPIC_API_KEY=athanor-local \
    ANTHROPIC_MODEL="$model" ANTHROPIC_DEFAULT_HAIKU_MODEL="$model" ANTHROPIC_SMALL_FAST_MODEL="$model" \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_TELEMETRY=1 DISABLE_AUTOUPDATER=1 DISABLE_ERROR_REPORTING=1 \
    API_TIMEOUT_MS=1800000 \
    timeout 3600 claude -p "$1" --output-format stream-json --verbose --permission-mode acceptEdits \
        --allowedTools Read Edit Write Glob Grep "Bash(python3:*)" "Bash(ls:*)" "Bash(cat:*)" "Bash(head:*)" \
        "Bash(grep:*)" "Bash(find:*)" "Bash(wc:*)" "Bash(git diff:*)" "Bash(git status:*)" < /dev/null
}

run_codex() {
    CODEX_HOME="$home/codex" ATHANOR_KEY=athanor-local \
    timeout 3600 codex exec --json --skip-git-repo-check --ephemeral -s workspace-write -C "$PWD" "$1" < /dev/null
}

run_opencode() {
    XDG_CONFIG_HOME="$home/opencode/config" XDG_DATA_HOME="$home/opencode/data" \
    XDG_CACHE_HOME="$home/opencode/cache" XDG_STATE_HOME="$home/opencode/state" \
    OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_SHARE=1 \
    timeout 3600 opencode run --format json --model "athanor/$model" "$1" < /dev/null
}

if [ "$which" = all ]; then
    tasks=$(cd "$here/tasks" && ls)
else
    [ -d "$here/tasks/$which" ] || { echo "no task $which" >&2; exit 2; }
    tasks=$which
fi

echo "# $version; $url; $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out/results.txt"
fails=0
for task in $tasks; do
    t="$here/tasks/$task"
    work=$(mktemp -d "${TMPDIR:-/tmp}/athanor-conformance-$harness-$task.XXXXXX")
    cp -R "$t/files/." "$work/"
    (cd "$work" && git init -q && git add -A && git -c user.name=conformance -c user.email=conformance@localhost commit -qm start)
    prompt=$(cat "$t/prompt.txt")
    start=$(date +%s)
    (cd "$work" && "run_$harness" "$prompt") > "$out/$harness-$task.jsonl" 2> "$out/$harness-$task.stderr"
    code=$?
    secs=$(( $(date +%s) - start ))
    if (cd "$work" && sh "$t/check.sh") > "$out/$harness-$task.check" 2>&1; then
        result=PASS
    else
        result=FAIL
        fails=$((fails + 1))
    fi
    (cd "$work" && git add -A && git -c core.pager=cat diff --cached --stat && git -c core.pager=cat diff --cached) \
        > "$out/$harness-$task.diff" 2>&1
    echo "$harness $task $result ${secs}s exit=$code $(head -c 200 "$out/$harness-$task.check" | tr '\n' ' ')" \
        | tee -a "$out/results.txt"
done
exit $fails
