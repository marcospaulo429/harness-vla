#!/usr/bin/env bash

set -euo pipefail

min_free_mib="${HARNESS_MIN_GPU_FREE_MIB:-16000}"
max_utilization="${HARNESS_MAX_GPU_UTILIZATION:-20}"

if ! [[ "$min_free_mib" =~ ^[0-9]+$ ]] || ! [[ "$max_utilization" =~ ^[0-9]+$ ]]; then
    echo "GPU thresholds must be non-negative integers" >&2
    exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required to select a GPU" >&2
    exit 2
fi

selected_gpu="$({
    nvidia-smi \
        --query-gpu=index,memory.free,utilization.gpu \
        --format=csv,noheader,nounits
} | awk -F',' -v min_free="$min_free_mib" -v max_util="$max_utilization" '
    {
            gpu_id = $1 + 0
        free_mib = $2 + 0
        utilization = $3 + 0
        if (free_mib >= min_free && utilization <= max_util) {
                print free_mib, utilization, gpu_id
        }
    }
' | sort -k1,1nr -k2,2n -k3,3n | awk 'NR == 1 { print $3 }')"

if [[ -z "$selected_gpu" ]]; then
    echo "No GPU has at least ${min_free_mib} MiB free and utilization <= ${max_utilization}%" >&2
    exit 1
fi

printf '%s\n' "$selected_gpu"