# A/B on Qwen3.5-0.8B decode, 4 rounds: step 13 (main), step 14, step 14 with
# the weights in 256 MiB blocks again (the engine's change reverted)
cd /home/ciarpa/Desktop/github/projects/athanor
R=docs/results/2026-11-26-step14-long-context/bench/ab
for round in 1 2 3 4; do
  for b in step13 step14 step14-blocks; do
    case $b in step13) bin=$MAIN;; step14) bin=./build/cuda/athanor;; *) bin=$NOALLOC;; esac
    $bin bench models/Qwen3.5-0.8B -p 0 -n 128 -d 0 -r 10 --no-hash --json $R/tg-0.8b-$b-$round.json > $R/tg-0.8b-$b-$round.log 2>&1
  done
done
