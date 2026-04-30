#!/usr/bin/env bash
set -euo pipefail

FOUNDATIONPOSE_REPO="${FOUNDATIONPOSE_REPO:-/home/tylerlum/github_repos/FoundationPose}"
CONDA_EXE="${CONDA_EXE:-/home/tylerlum/miniforge3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-foundationpose}"
DATA_ROOT="${DATA_ROOT:-/juno/u/yufeid/project_storage/shared_folder/input_data/gt_depth/gt_depth_0323}"
MESH_ROOT="${MESH_ROOT:-/home/tylerlum/github_repos/sam-3d-objects/2026-04-30_mesh_outputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FOUNDATIONPOSE_REPO}/2026-04-30_pose_outputs}"
TASK_GLOB="${TASK_GLOB:-*}"
MASK_GLOB="${MASK_GLOB:-*masks*}"
MAX_INITIAL_CANDIDATES="${MAX_INITIAL_CANDIDATES:-48}"
DEBUG="${DEBUG:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
    cat <<EOF
Usage: $0 [options]

Run FoundationPose for every generated SAM3D mesh under MESH_ROOT.
Inputs are read-only. Pose JSON/debug videos are written under OUTPUT_ROOT.

Options:
  --data-root PATH       Root containing task folders with rgb/, depth/, cam_K.txt, masks
  --mesh-root PATH       Local mesh output root from run_sam3_mesh_batch.sh
  --output-root PATH     Local output root for pose JSONs
  --task-glob GLOB       Task folder glob under DATA_ROOT, default: *
  --mask-glob GLOB       Mask folder glob inside each task, default: *masks*
  --debug N              FoundationPose debug level, default: 0. Use 1 to write vis.mp4.
  --max-candidates N     Initial pose candidate cap, default: 48
  --no-skip-existing     Re-run tasks even if poses.json already exists
  --dry-run              Print planned jobs without running FoundationPose
  -h, --help             Show this help

Example:
  $0 \\
    --data-root /juno/u/yufeid/project_storage/shared_folder/input_data/gt_depth/gt_depth_0323 \\
    --mesh-root /home/tylerlum/github_repos/sam-3d-objects/2026-04-30_mesh_outputs \\
    --output-root /home/tylerlum/github_repos/FoundationPose/2026-04-30_pose_outputs \\
    --debug 1
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --mesh-root)
            MESH_ROOT="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --task-glob)
            TASK_GLOB="$2"
            shift 2
            ;;
        --mask-glob)
            MASK_GLOB="$2"
            shift 2
            ;;
        --debug)
            DEBUG="$2"
            shift 2
            ;;
        --max-candidates)
            MAX_INITIAL_CANDIDATES="$2"
            shift 2
            ;;
        --no-skip-existing)
            SKIP_EXISTING=0
            shift
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

print_step() {
    echo
    echo "########################################################################"
    echo "$1"
    echo "########################################################################"
}

foundationpose() {
    cd "${FOUNDATIONPOSE_REPO}"
    eval "$("${CONDA_EXE}" shell.bash hook)"
    conda activate "${CONDA_ENV}"
    export PYTHONPATH="${FOUNDATIONPOSE_REPO}:${PYTHONPATH:-}"
    "$@"
}

sanitize_name() {
    local raw="$1"
    raw="${raw##*masks}"
    raw="${raw#_}"
    raw="${raw#-}"
    raw="${raw#.}"
    raw="${raw//\*/}"
    raw="${raw#_}"
    raw="${raw#-}"
    raw="${raw#.}"
    raw="${raw%.}"
    if [[ -z "${raw}" ]]; then
        raw="object"
    fi
    printf '%s' "${raw}" | tr -cs '[:alnum:]_-' '_'
}

find_mask_dir_for_object() {
    local task_dir="$1"
    local object_name="$2"
    local masks_dir base sanitized

    shopt -s nullglob
    for masks_dir in "${task_dir}"/${MASK_GLOB}; do
        [[ -d "${masks_dir}" ]] || continue
        base="$(basename "${masks_dir}")"
        [[ "${base}" == *tracking_results* ]] && continue
        [[ "${base}" == *robot_arm* ]] && continue
        sanitized="$(sanitize_name "${base}")"
        if [[ "${sanitized}" == "${object_name}" ]]; then
            printf '%s\n' "${masks_dir}"
            return 0
        fi
    done

    return 1
}

first_mask_file() {
    local masks_dir="$1"
    find "${masks_dir}" -maxdepth 1 -type f -name '*.png' | sort | head -1
}

run_one() {
    local task_name="$1"
    local object_name="$2"
    local mesh_path="$3"

    local task_dir masks_dir mask_path output_dir output_path
    task_dir="${DATA_ROOT}/${task_name}"
    output_dir="${OUTPUT_ROOT}/${task_name}/${object_name}"
    output_path="${output_dir}/poses.json"

    if [[ ! -d "${task_dir}/rgb" || ! -d "${task_dir}/depth" || ! -f "${task_dir}/cam_K.txt" ]]; then
        echo "Skipping missing/incomplete source task: ${task_dir}" >&2
        return
    fi

    if ! masks_dir="$(find_mask_dir_for_object "${task_dir}" "${object_name}")"; then
        echo "Skipping ${task_name}/${object_name}: could not find matching mask folder" >&2
        return
    fi

    mask_path="$(first_mask_file "${masks_dir}")"
    if [[ -z "${mask_path}" ]]; then
        echo "Skipping ${task_name}/${object_name}: no mask pngs in ${masks_dir}" >&2
        return
    fi

    if [[ "${SKIP_EXISTING}" == "1" && -f "${output_path}" ]]; then
        echo "Skipping existing poses: ${output_path}"
        return
    fi

    echo "Task: ${task_name}"
    echo "Object: ${object_name}"
    echo "Video dir: ${task_dir}"
    echo "Mesh: ${mesh_path}"
    echo "Mask: ${mask_path}"
    echo "Output: ${output_path}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        return
    fi

    mkdir -p "${output_dir}"
    foundationpose python extract_poses.py \
        --video_dir "${task_dir}" \
        --mesh_path "${mesh_path}" \
        --mask_path "${mask_path}" \
        --output_path "${output_path}" \
        --max_initial_candidates "${MAX_INITIAL_CANDIDATES}" \
        --debug "${DEBUG}"
}

print_step "Checking FoundationPose environment"
foundationpose python - <<'PY'
import cv2
import numpy
import trimesh
from Utils import set_logging_format
print("FoundationPose env OK")
PY

print_step "Discovering pose tracking jobs"
shopt -s nullglob
job_count=0
for task_dir in "${DATA_ROOT}"/${TASK_GLOB}; do
    [[ -d "${task_dir}" ]] || continue
    task_name="$(basename "${task_dir}")"
    mesh_task_dir="${MESH_ROOT}/${task_name}"
    [[ -d "${mesh_task_dir}" ]] || {
        echo "Skipping ${task_name}: no mesh task dir at ${mesh_task_dir}"
        continue
    }

    for object_dir in "${mesh_task_dir}"/*; do
        [[ -d "${object_dir}" ]] || continue
        object_name="$(basename "${object_dir}")"
        mesh_path="${object_dir}/mesh/mesh.obj"
        [[ -f "${mesh_path}" ]] || {
            echo "Skipping ${task_name}/${object_name}: missing ${mesh_path}"
            continue
        }

        job_count=$((job_count + 1))
        print_step "Job ${job_count}"
        run_one "${task_name}" "${object_name}" "${mesh_path}"
    done
done

print_step "Done"
echo "Processed ${job_count} pose tracking job(s)."
echo "Pose output root: ${OUTPUT_ROOT}"
