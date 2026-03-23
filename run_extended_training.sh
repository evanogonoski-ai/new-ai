#!/bin/bash
# Extended Training: All 4 models at 10,000 steps (sequential)
# Estimated total time: ~16.5 hours

set -e
cd /home/user/new-ai

LOG_DIR=logs
SCRIPT=neurogenesis_v2/scripts/train_fair_fight.py

echo "$(date): Starting extended training run (10K steps, sequential)" | tee -a $LOG_DIR/extended_run.log

# Model A: Baseline (4 layers, ~16.9M params) — ~4.0 hours
echo "$(date): Starting Model A (Baseline)..." | tee -a $LOG_DIR/extended_run.log
python $SCRIPT \
  --model baseline \
  --steps 10000 \
  --checkpoint-every 1000 \
  --log-prefix extended \
  --checkpoint-subdir 10k \
  2>&1 | tee $LOG_DIR/extended_baseline_stdout.txt
echo "$(date): Model A done." | tee -a $LOG_DIR/extended_run.log

# Model D: Universal Transformer (4 iter, no gating) — ~3.0 hours
echo "$(date): Starting Model D (Universal)..." | tee -a $LOG_DIR/extended_run.log
python $SCRIPT \
  --model universal \
  --n-iterations 4 \
  --steps 10000 \
  --checkpoint-every 1000 \
  --log-prefix extended \
  --checkpoint-subdir 10k \
  2>&1 | tee $LOG_DIR/extended_universal_stdout.txt
echo "$(date): Model D done." | tee -a $LOG_DIR/extended_run.log

# Model B: Resonance 4-iter — ~3.3 hours
echo "$(date): Starting Model B (Resonance 4-iter)..." | tee -a $LOG_DIR/extended_run.log
python $SCRIPT \
  --model resonance \
  --n-iterations 4 \
  --steps 10000 \
  --checkpoint-every 1000 \
  --log-prefix extended \
  --checkpoint-subdir 10k \
  2>&1 | tee $LOG_DIR/extended_resonance_4iter_stdout.txt
echo "$(date): Model B done." | tee -a $LOG_DIR/extended_run.log

# Model C: Resonance 8-iter — ~6.2 hours
echo "$(date): Starting Model C (Resonance 8-iter)..." | tee -a $LOG_DIR/extended_run.log
python $SCRIPT \
  --model resonance \
  --n-iterations 8 \
  --steps 10000 \
  --checkpoint-every 1000 \
  --log-prefix extended \
  --checkpoint-subdir 10k \
  2>&1 | tee $LOG_DIR/extended_resonance_8iter_stdout.txt
echo "$(date): Model C done." | tee -a $LOG_DIR/extended_run.log

echo "$(date): ALL 4 MODELS COMPLETE!" | tee -a $LOG_DIR/extended_run.log
echo "Check logs/extended_*_eval.txt for results." | tee -a $LOG_DIR/extended_run.log
