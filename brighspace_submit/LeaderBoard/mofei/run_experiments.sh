#!/bin/bash
# ============================================================================
# REUSABLE EXPERIMENT SUITE
# ============================================================================
# Usage:
#   bash run_experiments.sh                        # defaults (unet_v3)
#   bash run_experiments.sh --model unet           # original unet
#   bash run_experiments.sh --model unet_v3 --epochs 200 --wd 5e-4
#   bash run_experiments.sh --skip-phase1          # skip regularization comparison
#   bash run_experiments.sh --folds-only           # only train ensemble folds + submit
# ============================================================================

set -o pipefail
# No set -e: we want to continue past individual failures

# ─── Defaults (override via CLI) ─────────────────────────────────────────────
MODEL="unet_v3"
EPOCHS=150
BS=64
LR=1e-4
WD=1e-3
SSIM_W=0.4
L1_W=0.3
EDGE_W=0.15
FFT_W=0.15
N_FOLDS=5
FOLD_SIZE=2
SKIP_PHASE1=false
FOLDS_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2;;
        --epochs) EPOCHS="$2"; shift 2;;
        --bs) BS="$2"; shift 2;;
        --lr) LR="$2"; shift 2;;
        --wd) WD="$2"; shift 2;;
        --ssim_w) SSIM_W="$2"; shift 2;;
        --l1_w) L1_W="$2"; shift 2;;
        --edge_w) EDGE_W="$2"; shift 2;;
        --fft_w) FFT_W="$2"; shift 2;;
        --skip-phase1) SKIP_PHASE1=true; shift;;
        --folds-only) FOLDS_ONLY=true; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

LOG="experiment_log_${MODEL}.txt"
LOSS_ARGS="--ssim_w $SSIM_W --l1_w $L1_W --edge_w $EDGE_W --fft_w $FFT_W"

echo "========================================" | tee $LOG
echo "EXPERIMENT SUITE — $(date)" | tee -a $LOG
echo "Model=$MODEL, Epochs=$EPOCHS, BS=$BS, LR=$LR, WD=$WD" | tee -a $LOG
echo "Loss: SSIM=$SSIM_W L1=$L1_W Edge=$EDGE_W FFT=$FFT_W" | tee -a $LOG
echo "========================================" | tee -a $LOG

# ─── Helper: train + eval a single fold ──────────────────────────────────────
train_fold() {
    local FOLD_IDX=$1
    local FOLD_NAME=$2
    local FOLD_LOG="runs/train_${MODEL}_${FOLD_NAME}.log"

    echo "[$(date +%H:%M)] Training $MODEL $FOLD_NAME (val_start=$FOLD_IDX)..." | tee -a $LOG
    if python3 train.py --model $MODEL --data_dir ./data --epochs $EPOCHS \
        --batch_size $BS --lr $LR --val_subjects $FOLD_SIZE --val_start $FOLD_IDX \
        --weight_decay $WD $LOSS_ARGS 2>&1 | tee $FOLD_LOG | tail -5 | tee -a $LOG; then

        echo "[$(date +%H:%M)] Evaluating $FOLD_NAME on holdout..." | tee -a $LOG
        python3 eval_fold.py --model $MODEL --checkpoint runs/${MODEL}_split${FOLD_IDX}_best.pt \
            --data_dir ./data --val_start $FOLD_IDX --val_subjects $FOLD_SIZE 2>&1 | tee -a $LOG
    else
        echo "  *** TRAINING FAILED for $FOLD_NAME — see $FOLD_LOG ***" | tee -a $LOG
    fi
}

# ─── Phase 1: Regularization Comparison (optional) ───────────────────────────
if [ "$SKIP_PHASE1" = false ] && [ "$FOLDS_ONLY" = false ]; then
    echo "" | tee -a $LOG
    echo "=== PHASE 1: REGULARIZATION COMPARISON (holdout=16,17) ===" | tee -a $LOG

    echo "[$(date +%H:%M)] Baseline: unet (no dropout), wd=$WD" | tee -a $LOG
    python3 train.py --model unet --data_dir ./data --epochs $EPOCHS \
        --batch_size $BS --lr $LR --val_subjects 2 --val_start 16 \
        --weight_decay $WD $LOSS_ARGS 2>&1 | tee runs/train_unet_baseline.log | tail -5 | tee -a $LOG
    echo "Eval:" | tee -a $LOG
    python3 eval_fold.py --model unet --checkpoint runs/unet_split16_best.pt \
        --data_dir ./data --val_start 16 2>&1 | tee -a $LOG

    echo "[$(date +%H:%M)] Test: $MODEL, wd=$WD" | tee -a $LOG
    python3 train.py --model $MODEL --data_dir ./data --epochs $EPOCHS \
        --batch_size $BS --lr $LR --val_subjects 2 --val_start 16 \
        --weight_decay $WD $LOSS_ARGS 2>&1 | tee runs/train_${MODEL}_phase1.log | tail -5 | tee -a $LOG
    cp runs/${MODEL}_split16_best.pt runs/${MODEL}_phase1_best.pt 2>/dev/null || true
    echo "Eval:" | tee -a $LOG
    python3 eval_fold.py --model $MODEL --checkpoint runs/${MODEL}_phase1_best.pt \
        --data_dir ./data --val_start 16 2>&1 | tee -a $LOG
fi

# ─── Phase 2: N-Fold Ensemble ────────────────────────────────────────────────
echo "" | tee -a $LOG
echo "=== PHASE 2: ${N_FOLDS}-FOLD ENSEMBLE ===" | tee -a $LOG

N_SUBJECTS=18
STEP=$(( (N_SUBJECTS - FOLD_SIZE) / (N_FOLDS - 1) ))
FOLD_STARTS=()
for ((i=0; i<N_FOLDS; i++)); do
    IDX=$((i * STEP))
    if [ $IDX -gt $((N_SUBJECTS - FOLD_SIZE)) ]; then
        IDX=$((N_SUBJECTS - FOLD_SIZE))
    fi
    FOLD_STARTS+=($IDX)
done

echo "Fold start indices: ${FOLD_STARTS[*]}" | tee -a $LOG

for i in "${!FOLD_STARTS[@]}"; do
    IDX=${FOLD_STARTS[$i]}

    # Reuse Phase 1 checkpoint if it matches
    if [ "$SKIP_PHASE1" = false ] && [ "$FOLDS_ONLY" = false ] && [ "$IDX" = "16" ]; then
        echo "[Fold $i] Reusing Phase 1 checkpoint (split16)" | tee -a $LOG
        cp runs/${MODEL}_phase1_best.pt runs/${MODEL}_split${IDX}_best.pt 2>/dev/null || true
        python3 eval_fold.py --model $MODEL --checkpoint runs/${MODEL}_split${IDX}_best.pt \
            --data_dir ./data --val_start $IDX --val_subjects $FOLD_SIZE 2>&1 | tee -a $LOG
        continue
    fi

    train_fold $IDX "fold$i"
done

# ─── Phase 3: All-Data Model ─────────────────────────────────────────────────
if [ "$FOLDS_ONLY" = false ]; then
    echo "" | tee -a $LOG
    echo "=== PHASE 3: ALL-DATA MODEL ===" | tee -a $LOG
    echo "[$(date +%H:%M)] Training $MODEL on all 18 subjects..." | tee -a $LOG
    python3 train.py --model $MODEL --data_dir ./data --epochs $EPOCHS \
        --batch_size $BS --lr $LR --val_subjects 0 \
        --weight_decay $WD $LOSS_ARGS 2>&1 | tee runs/train_${MODEL}_alldata.log | tail -5 | tee -a $LOG
fi

# ─── Phase 4: Generate Submissions ───────────────────────────────────────────
echo "" | tee -a $LOG
echo "=== PHASE 4: GENERATE SUBMISSIONS ===" | tee -a $LOG

if [ "$FOLDS_ONLY" = false ]; then
    echo "[$(date +%H:%M)] Single-model submission..." | tee -a $LOG
    python3 infer.py --model $MODEL --checkpoint runs/${MODEL}_final.pt \
        --data_dir ./data --output submission_single_${MODEL}.csv 2>&1 | tee -a $LOG
fi

ENSEMBLE_ARGS=""
for IDX in "${FOLD_STARTS[@]}"; do
    ENSEMBLE_ARGS="$ENSEMBLE_ARGS ${MODEL}:runs/${MODEL}_split${IDX}_best.pt"
done

echo "[$(date +%H:%M)] Ensemble submission (${N_FOLDS} folds)..." | tee -a $LOG
python3 ensemble_infer.py --data_dir ./data \
    --output submission_ensemble_${MODEL}.csv \
    --models $ENSEMBLE_ARGS 2>&1 | tee -a $LOG

# ─── Summary ──────────────────────────────────────────────────────────────────
echo "" | tee -a $LOG
echo "========================================" | tee -a $LOG
echo "ALL DONE — $(date)" | tee -a $LOG
echo "Submissions:" | tee -a $LOG
[ "$FOLDS_ONLY" = false ] && echo "  submission_single_${MODEL}.csv" | tee -a $LOG
echo "  submission_ensemble_${MODEL}.csv" | tee -a $LOG
echo "========================================" | tee -a $LOG
echo ""
echo "Review: cat $LOG"
