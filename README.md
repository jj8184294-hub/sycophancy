python refactor_single.py \
  --claims claims_200.jsonl \
  --out_dir results/run_200_default_last3 \
  --prompt_set default \
  --patch_last_n 3 \
  --component_layers 5 \
  --topk_neurons 200 \
  --timing_every 10

python remove_pressure.py --mode fit   --claims claims_200.jsonl   --out_dir out_vec   --layer 19 --hook resid_post --pos last   --device cpu --dtype bfloat16

python remove_pressure.py --mode eval \
  --claims claims_200.jsonl  \
  --out_dir out_eval \
  --vec_path out_vec/pressure_vec.pt \
  --layer 19 --hook resid_post --pos last \
  --alphas 0.0,0.25,0.5,0.75,1.0 \
  --device cpu --dtype bfloat16