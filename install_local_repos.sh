#!/usr/bin/env bash
# Install sibling repos in editable mode. Skips siblings that are not cloned.
# Required for examples/verify: multi_channel_dataset_creation
# Optional: ML_geo_production, ML_Production

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(cd "${SCRIPT_DIR}/.." && pwd)"

install_sibling() {
  local name="$1"
  local dir="${PARENT}/${name}"
  if [[ -d "${dir}" ]]; then
    echo "INSTALL_LOCAL: ${name}"
    (cd "${dir}" && pip install -e .)
  else
    echo "INSTALL_LOCAL: skip ${name} (not found at ${dir})"
  fi
}

install_sibling ML_geo_production
install_sibling multi_channel_dataset_creation
install_sibling ML_Production

echo "INSTALL_LOCAL: ML_sdfi_fastai2"
(cd "${SCRIPT_DIR}" && pip install -e .)
