#!/bin/bash
set -euo pipefail

ROOT_DIR="/myriadfs/home/zcemexx/projects/MREPT_code"
LABEL_SCRIPT="$ROOT_DIR/sub_array.sh"
REPLOT_SCRIPT="$ROOT_DIR/sub_replot.sh"

label_job_id=$(qsub -terse "$LABEL_SCRIPT")
echo "Submitted label array job: $label_job_id"

replot_job_id=$(qsub -terse -hold_jid "$label_job_id" "$REPLOT_SCRIPT")
echo "Submitted replot job: $replot_job_id (hold_jid=$label_job_id)"
