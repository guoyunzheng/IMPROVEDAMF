#!/usr/bin/env bash
# Reproduce the ITT runtime repair verified with PyTorch 2.5.1 / CUDA 12.4.
# Run after activating the environment created by conda_environment_base.yaml.
set -euo pipefail

if [[ "$(uname -s)" != Linux ]]; then
    printf '%s\n' 'This runtime repair is for Linux environments.' >&2
    exit 1
fi
: "${CONDA_PREFIX:?Activate the target Conda environment first.}"

python - <<'PY'
import os
import sys
from pathlib import Path

if Path(sys.prefix).resolve() != Path(os.environ['CONDA_PREFIX']).resolve():
    raise SystemExit('Python does not belong to the active Conda environment.')
PY

itt_shared="$CONDA_PREFIX/lib/libittnotify.so"
itt_static="$CONDA_PREFIX/lib/libittnotify.a"

if [[ ! -e "$itt_shared" ]]; then
    command -v gcc >/dev/null || {
        printf '%s\n' 'gcc is required to link the ITT runtime.' >&2
        exit 1
    }
    if [[ ! -f "$itt_static" ]]; then
        printf '%s\n' 'Missing libittnotify.a; install ittapi-static=3.28.0 first.' >&2
        exit 1
    fi

    itt_build_output="$(mktemp "$CONDA_PREFIX/lib/.libittnotify.XXXXXX.so")"
    trap 'rm -f -- "$itt_build_output"' EXIT
    gcc -shared \
        -Wl,-soname,libittnotify.so \
        -Wl,--whole-archive "$itt_static" \
        -Wl,--no-whole-archive \
        -ldl -pthread \
        -o "$itt_build_output"
    chmod 755 "$itt_build_output"
    mv -- "$itt_build_output" "$itt_shared"
    trap - EXIT
    printf 'Created %s\n' "$itt_shared"
else
    printf 'Using existing %s\n' "$itt_shared"
fi

python - <<'PY'
import torch

print('PyTorch import: PASS')
print('PyTorch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
PY
