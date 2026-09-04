#!/bin/bash
DEC="$1"; ENC="$2"; LABEL="$3"; shift 3
echo "############ $LABEL (integrated C++, single compact round/process) ############"
for spec in "$@"; do
  name="${spec%%:*}"; mem="${spec##*:}"
  # 5 process launches; filler REPS=4 (stable), compact CREPS=1 (stable)
  fills=(); comps=(); infers=(); cmpcts=(); shrink=""; corr=""
  for i in 1 2 3 4 5; do
    out=$(./build/integ "$DEC" "$ENC" "$mem" 1 4 1 2>/dev/null)
    [ -z "$out" ] && { echo "$name: CRASH on launch $i"; break; }
    f=$(echo "$out" | grep 'FILLER'  | grep -oE '=[0-9.]+ ms' | head -1 | tr -dc '0-9.')
    c=$(echo "$out" | grep 'COMPACT round' | grep -oE 'round_wall\(med\)=[0-9.]+' | tr -dc '0-9.')
    ii=$(echo "$out" | grep -oE 'infer\(med\)=[0-9.]+' | tr -dc '0-9.')
    cc=$(echo "$out" | grep -oE 'compaction\(med\)=[0-9.]+' | tr -dc '0-9.')
    fills+=($f); comps+=($c); infers+=($ii); cmpcts+=($cc)
    shrink=$(echo "$out" | grep 'shrinks(widths)' | sed 's/.*=//')
    corr=$(echo "$out" | grep 'correctness')
  done
  med(){ printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
  echo "$name  mem=$mem  shrinks=[$shrink]"
  echo "   FILLER  wall=$(med "${fills[@]}") ms | COMPACT wall=$(med "${comps[@]}") ms (infer=$(med "${infers[@]}") compaction=$(med "${cmpcts[@]}"))"
  fm=$(med "${fills[@]}"); cm=$(med "${comps[@]}")
  echo "   net(COMPACT-FILLER)=$(awk "BEGIN{printf \"%.1f\", $cm-$fm}") ms  $( [ "$corr" ] && echo "| $corr" )"
done
