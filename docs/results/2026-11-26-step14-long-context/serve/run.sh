cd /home/ciarpa/Desktop/github/projects/athanor
L=$CLAUDE_JOB_DIR/tmp/long
R=docs/results/2026-11-26-step14-long-context/serve
mkdir -p $R
SNAP=$L/snaps
for run in 1 2; do
  ./build/cuda/athanor serve models/Qwen3.5-9B --ctx 131072 --chunk 1024 --no-think --snapshots $SNAP --port 8091 > $R/serve$run.log 2>&1 &
  pid=$!
  until grep -q "listening" $R/serve$run.log 2>/dev/null || ! kill -0 $pid 2>/dev/null; do sleep 2; done
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > $R/vram$run.txt
  t0=$(date +%s.%N)
  curl -s http://127.0.0.1:8091/v1/chat/completions -H 'content-type: application/json' -d @$L/sys$run.json > $R/response$run.json
  t1=$(date +%s.%N)
  echo "request $run: $(echo "$t1 - $t0" | bc) s" >> $R/times.txt
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader >> $R/vram$run.txt
  kill -INT $pid; wait $pid
  ls -la $SNAP > $R/snapshots-after-run$run.txt
done
