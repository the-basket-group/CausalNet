#!/usr/bin/env bash
# CausalNet phase-attention experiment (RQ1).
#
# Question: does disentangling the per-phase spatial attention in the Causal Attention
# Block (each phase gets its own parameters) beat CausalNet's shared-attention baseline,
# beyond the added capacity? Run from inside CausalNet/ on the VM; the optical-flow
# datasets are bundled in ./datasets, so there is no preprocessing step here.
#
#   shared        - one spatial-attention set for both phases (upstream default == baseline)
#   disentangled  - per-phase parameters (the contribution)
#   shared_wide   - shared, but width-matched to disentangled (capacity control)
#
# shared @ seed 2025 reproduces the upstream baseline (its hardcoded seed was 2025).
set -e

PY="python"
SEEDS="${SEEDS:-2025 2026 2027}"   # override for a quick pass: SEEDS="2025" ./run_all.sh
MODES="${MODES:-shared disentangled shared_wide}"

# 1. RQ1: train each variant per seed, then score its run folder (combined + per-dataset).
for s in $SEEDS; do
  for m in $MODES; do
    echo "=== attn_mode=$m seed=$s ==="
    RD="./results/${m}_seed${s}"
    $PY main_train.py --attn_mode "$m" --seed "$s"
    $PY calculate_all_results.py        --results_dir "$RD"   # combined 3-class
    $PY calculate_all_results_CASMEII.py --results_dir "$RD"
    $PY calculate_all_results_SAMM.py    --results_dir "$RD"
    $PY calculate_all_results_SMIC.py    --results_dir "$RD"
  done
done

echo "Done. Per-run metrics: ./results/<mode>_seed<seed>/CausalNet*.xlsx"
