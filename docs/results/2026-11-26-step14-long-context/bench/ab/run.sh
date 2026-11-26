# A/B: step 13's build (main, f840d33, in a worktree) against step 14's, alternately
cd /home/ciarpa/Desktop/github/projects/athanor
for round in 1 2; do
  for b in step13 step14; do
    bin=$([ $b = step13 ] && echo "$MAIN" || echo ./build/cuda/athanor)
    $bin bench models/Qwen3.5-0.8B -p 512 -n 128 -d 0 -r 5 --no-hash --json docs/results/2026-11-26-step14-long-context/bench/ab/0.8b-$b-$round.json > docs/results/2026-11-26-step14-long-context/bench/ab/0.8b-$b-$round.log 2>&1
    $bin bench models/Qwen3.6-35B-A3B --prompt-ids docs/results/2026-11-25-step12-moe/prompt-short.ids -n 128 -r 5 --warmup 1 --no-hash --json docs/results/2026-11-26-step14-long-context/bench/ab/35b-short-$b-$round.json > docs/results/2026-11-26-step14-long-context/bench/ab/35b-short-$b-$round.log 2>&1
  done
done
