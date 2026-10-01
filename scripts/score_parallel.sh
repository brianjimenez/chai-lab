#!/usr/bin/env bash
# Scores all the PDB files of a folder with `chai-lab score`, one process per GPU
# in parallel, and merges the results (sorted by aggregate score, best first).
#
# Usage: scripts/score_parallel.sh STRUCTURES_DIR OUTPUT_DIR [MSA_DIR] [N_GPUS]
#   N_GPUS defaults to the number of GPUs found by nvidia-smi.
#   Extra options for `chai-lab score` can be given in SCORE_ARGS, e.g.
#   SCORE_ARGS="--no-keep-models-on-gpu" scripts/score_parallel.sh ...
#
# Output: OUTPUT_DIR/scores.csv, plus OUTPUT_DIR/parts/ (per-GPU lists, CSVs and
# logs) and OUTPUT_DIR/timing.txt. An interrupted run continues where it stopped
# if the same command is run again (see the partial scores in `chai-lab score`).
set -euo pipefail

structures_dir=${1:?usage: $0 STRUCTURES_DIR OUTPUT_DIR [MSA_DIR] [N_GPUS]}
output_dir=${2:?usage: $0 STRUCTURES_DIR OUTPUT_DIR [MSA_DIR] [N_GPUS]}
msa_dir=${3:-}
n_gpus=${4:-$(nvidia-smi --list-gpus | wc -l)}
here=$(cd "$(dirname "$0")" && pwd)

parts=$output_dir/parts
mkdir -p "$parts"
rm -f "$parts"/gpu_*.list

# Deal the structures (skipping hidden macOS ._* files) round-robin to the GPUs
i=0
for pdb in "$structures_dir"/[!.]*.pdb; do
    echo "$pdb" >> "$parts/gpu_$((i % n_gpus)).list"
    i=$((i + 1))
done
echo "Scoring $i structures on $n_gpus GPUs"

start=$(date +%s)
pids=()
for ((g = 0; g < n_gpus; g++)); do
    [ -f "$parts/gpu_$g.list" ] || continue
    msa_args=()
    [ -n "$msa_dir" ] && msa_args=(--msa-directory "$msa_dir")
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$g chai-lab score $(cat "$parts/gpu_$g.list") \
        "$parts/gpu_$g.csv" "${msa_args[@]}" ${SCORE_ARGS:-} \
        > "$parts/gpu_$g.log" 2>&1 &
    pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
elapsed=$(($(date +%s) - start))
echo "Elapsed: ${elapsed} s" | tee "$output_dir/timing.txt"
if [ "$failed" -ne 0 ]; then
    echo "A GPU run failed: see $parts/gpu_*.log; run again to resume" >&2
    exit 1
fi

python "$here/merge_scores.py" "$output_dir/scores.csv" "$parts"/gpu_*.csv
