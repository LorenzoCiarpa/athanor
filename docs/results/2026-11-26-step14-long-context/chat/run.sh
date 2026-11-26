# athanor chat at 131072 tokens on Qwen3.5-9B, twice on the same snapshot folder.
# hay.txt: python3 ../serve/haystack.py 5600 text > hay.txt (one line: one message)
cd /home/ciarpa/Desktop/github/projects/athanor
R=docs/results/2026-11-26-step14-long-context/chat
for run in 1 2; do
  /usr/bin/time -v ./build/cuda/athanor chat models/Qwen3.5-9B --ctx 131072 --chunk 1024 --temp 0 --stats \
    --max-tokens 24 --snapshots "$SNAP" < "$HAY" > $R/answer$run.txt 2> $R/chat$run.log
done
