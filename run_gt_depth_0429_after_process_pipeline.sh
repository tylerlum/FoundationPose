#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/juno/u/yufeid/project_storage/shared_folder/input_data/gt_depth/gt_depth_0429/after_process}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/juno/u/tylerlum/2026-05-01_sam3d_foundationpose_gt_depth_0429_after_process}"
EXCLUDE_TASKS="${EXCLUDE_TASKS:-visionpro_12,visionpro_13}"
SAM3_BATCH="${SAM3_BATCH:-/home/tylerlum/github_repos/segment-anything-2-real-time/run_sam3_mesh_batch.sh}"
FOUNDATIONPOSE_BATCH="${FOUNDATIONPOSE_BATCH:-/home/tylerlum/github_repos/FoundationPose/run_foundationpose_batch.sh}"
DRY_RUN=0

usage() {
    cat <<EOF
Usage: $0 [options]

Run SAM3D mesh generation, then FoundationPose pose tracking, for gt_depth_0429/after_process.

Options:
  --data-root PATH       Dataset root, default: ${DATA_ROOT}
  --output-root PATH     Output root containing mesh_outputs/, pose_outputs/, logs/
                         default: ${OUTPUT_ROOT}
  --exclude-tasks LIST   Comma or space separated task names to skip
                         default: ${EXCLUDE_TASKS}
  --dry-run              Print planned jobs without running inference
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --exclude-tasks)
            EXCLUDE_TASKS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

MESH_ROOT="${OUTPUT_ROOT}/mesh_outputs"
POSE_ROOT="${OUTPUT_ROOT}/pose_outputs"
LOG_ROOT="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_ROOT}"

SAM3_ARGS=(
    --data-root "${DATA_ROOT}"
    --output-root "${MESH_ROOT}"
    --exclude-tasks "${EXCLUDE_TASKS}"
)

FOUNDATIONPOSE_ARGS=(
    --data-root "${DATA_ROOT}"
    --mesh-root "${MESH_ROOT}"
    --output-root "${POSE_ROOT}"
    --exclude-tasks "${EXCLUDE_TASKS}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
    SAM3_ARGS+=(--dry-run)
    FOUNDATIONPOSE_ARGS+=(--dry-run)
fi

echo "Data root: ${DATA_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Excluded tasks: ${EXCLUDE_TASKS}"
echo "Mesh root: ${MESH_ROOT}"
echo "Pose root: ${POSE_ROOT}"

set -o pipefail
"${SAM3_BATCH}" "${SAM3_ARGS[@]}" 2>&1 | tee "${LOG_ROOT}/sam3d_$(date +%Y%m%d_%H%M%S).log"
"${FOUNDATIONPOSE_BATCH}" "${FOUNDATIONPOSE_ARGS[@]}" 2>&1 | tee "${LOG_ROOT}/foundationpose_$(date +%Y%m%d_%H%M%S).log"
