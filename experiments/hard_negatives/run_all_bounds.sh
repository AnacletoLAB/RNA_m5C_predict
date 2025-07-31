#!/usr/bin/env bash
UPPERS=(0.5 0.6 0.7 0.8 0.9 1.0)
LOWERS=(0.0)

mkdir -p logs

for L in "${LOWERS[@]}"; do
  for U in "${UPPERS[@]}"; do
    echo "Launching lower=$L upper=$U (hard negatives ON)"
    python train.py --lower "$L" --upper "$U" --hard-negatives \
           > "logs/lower_${L}_upper_${U}_hard.log" 2>&1 &
  done
done

wait
echo "All jobs finished."
