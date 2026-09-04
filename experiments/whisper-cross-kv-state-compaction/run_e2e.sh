#!/bin/bash
# Orchestrate isolated A/B measurements per batch. arg1 = model dir, arg2 = label
MODEL="$1"; LABEL="$2"
PY="${PYTHON:-python3}"
FILT='grep -vE "Warning|warn|torch|FutureWarning|return |Setting|attention_mask|The attention|input name"'
declare -A BATCH=(
  [B2_imbal]="x12,short05"
  [B2_bal]="x2,tts2"
  [B4_stagger]="x12,x8,x2,short05"
  [B4_short]="x1,tts1,tts2,x2"
  [B8_mixed]="x12,x8,x6,x4,x3,x2,tts1,short05"
)
echo "###### $LABEL ######"
for bn in B2_imbal B2_bal B4_stagger B4_short B8_mixed; do
  m="${BATCH[$bn]}"
  fa=$($PY /tmp/cand3_exp/e2e.py "$MODEL" one filler  "$m" 2>/dev/null | grep '^JSON' | sed 's/^JSON//')
  fb=$($PY /tmp/cand3_exp/e2e.py "$MODEL" one compact "$m" 2>/dev/null | grep '^JSON' | sed 's/^JSON//')
  echo "BATCH $bn members=$m"
  echo "  FILLER  $fa"
  echo "  COMPACT $fb"
done
