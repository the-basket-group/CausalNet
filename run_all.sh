#!/usr/bin/env bash
# CausalNet phase-attention experiment (RQ1).
#
# Question: does separating the per-phase spatial attention in the Causal Attention
# Block (each phase gets its own parameters) beat CausalNet's shared-attention baseline,
# beyond the added capacity? Run from inside CausalNet/ on the VM; the optical-flow
# datasets are bundled in ./datasets, so there is no preprocessing step here.
#
#   shared        - one spatial-attention set for both phases (upstream default == baseline)
#   dual  - per-phase parameters (the contribution)
#   shared_matched   - shared, but capacity-matched to dual (capacity control)
#
# shared @ seed 2025 reproduces the upstream baseline (its hardcoded seed was 2025).
set -e

PY="python"
SEEDS="${SEEDS:-2025 2026 2027 2028 2029}"   # 5 seeds; override for a quick pass: SEEDS="2025" ./run_all.sh
MODES="${MODES:-shared dual shared_matched}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-200}"   # patience >= EPOCHS never fires, so nothing stops early;
                              # the best-validation checkpoint is still the one tested.
TAG="${TAG:-nostop}"          # results go to ./results/$TAG, so a re-run cannot collide
                              # with an earlier one (main_train.py appends to *_acc.txt).

OUT="./results/${TAG}"
echo "epochs=$EPOCHS  patience=$PATIENCE  ->  $OUT"

# 1. RQ1: train each variant per seed, then score its run folder (combined + per-dataset).
for s in $SEEDS; do
  for m in $MODES; do
    echo "=== attn_mode=$m seed=$s ==="
    RD="${OUT}/${m}_seed${s}"
    $PY main_train.py --attn_mode "$m" --seed "$s" \
        --epochs "$EPOCHS" --patience "$PATIENCE" --run_name "${TAG}/${m}_seed${s}"
    $PY calculate_all_results.py        --results_dir "$RD"   # combined 3-class
    $PY calculate_all_results_CASMEII.py --results_dir "$RD"
    $PY calculate_all_results_SAMM.py    --results_dir "$RD"
    $PY calculate_all_results_SMIC.py    --results_dir "$RD"
  done
done

# 2. Aggregate across seeds (mean +/- std) and plot loss / val-UF1 curves per mode.
$PY aggregate_results.py --results_dir "$OUT" --modes $MODES --seeds $SEEDS --curves

echo "Done. Per-run metrics: ./results/<mode>_seed<seed>/CausalNet*.xlsx"
echo "Aggregate table: printed above | curves: ./results/history_<mode>.png"
