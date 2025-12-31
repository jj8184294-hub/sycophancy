python refactor_single.py \
  --claims claims_200.jsonl \
  --out_dir results/run_200_default_last3 \
  --prompt_set default \
  --patch_last_n 3 \
  --component_layers 5 \
  --topk_neurons 200 \
  --timing_every 10