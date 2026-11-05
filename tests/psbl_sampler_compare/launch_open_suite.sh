#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/jlu/code/python/bagle"
OUT="${ROOT}/tests/psbl_sampler_compare/runs_open"
LOG="${OUT}/run.log"
PIDFILE="${OUT}/run.pid"
export MPLCONFIGDIR="${ROOT}/tests/psbl_sampler_compare/.mplconfig"
mkdir -p "${OUT}" "${MPLCONFIGDIR}"
cd "${ROOT}/tests/psbl_sampler_compare"
setsid -f zsh -lic "
  source /opt/miniforge3/etc/profile.d/conda.sh
  conda activate /Users/jlu/.conda/envs/py314
  cd '${ROOT}/tests/psbl_sampler_compare'
  export MPLCONFIGDIR='${MPLCONFIGDIR}'
  export PYTHONPATH='${ROOT}/src:.'
  exec python -u run_comparison.py --outdir runs_open --prior-mode open --scenario all --model-cadence 10
" > "${LOG}" 2>&1
# setsid -f returns immediately; find the python pid
sleep 2
pgrep -nf 'run_comparison.py --outdir runs_open' > "${PIDFILE}" || true
echo "Launched PID $(cat "${PIDFILE}" 2>/dev/null || echo unknown)"
echo "Log: ${LOG}"
