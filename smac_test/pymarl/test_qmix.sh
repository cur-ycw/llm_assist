#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pymarl

cd /root/ycw/smac_test/pymarl
pkill -9 -f SC2_x64 2>/dev/null || true
sleep 2

echo "=== Testing QMIX with SC2.4.10 ==="
echo "Date: $(date)"
echo ""

python src/main.py \
    --config=qmix \
    --env-config=sc2 \
    with \
    t_max=20000 \
    save_model=False \
    use_cuda=True \
    2>&1 | tee /tmp/qmix_test.log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=== Exit code: $EXIT_CODE ==="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ QMIX test PASSED"
else
    echo "❌ QMIX test FAILED"
    tail -30 /tmp/qmix_test.log
fi

pkill -9 -f SC2_x64 2>/dev/null || true
exit $EXIT_CODE
