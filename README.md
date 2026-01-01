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

python remove_pressure.py sweep \
  --claims claims_200.jsonl \
  --out_dir out_alpha_sweep_small \
  --vec_path out_vec/pressure_vec.pt \
  --baseline_csv out_localize/per_claim.csv \
  --alphas 0.0,0.05,0.1,0.15,0.2,0.25 \
  --subset_n 20 --seed 0 \
  --layer 19 --hook resid_post --pos last \
  --threads 8 --device cpu --dtype float32

python remove_pressure.py final \
  --claims data/claims.jsonl \
  --out_dir out_eval_200_alpha015 \
  --vec_path out_vec/pressure_vec.pt \
  --baseline_csv out_localize/per_claim.csv \
  --alpha 0.15 \
  --layer 19 --hook resid_post --pos last \
  --threads 8 --device cpu --dtype float32

python remove_pressure.py final \
  --claims data/new_claims.jsonl \
  --out_dir out_eval_new_alpha015 \
  --vec_path out_vec/pressure_vec.pt \
  --alpha 0.15 \
  --layer 19 --hook resid_post --pos last \
  --threads 8 --device cpu --dtype float32

python remove_pressure_stronger.py fit_ridge \
  --claims claims_200.jsonl \
  --baseline_csv per_claim.csv \
  --out_dir out_ridge \
  --layer 19 --hook resid_post --pos last \
  --lam 10.0 \
  --device cpu --dtype float32 --threads 8 \
  --subset_n 200 --seed 0
