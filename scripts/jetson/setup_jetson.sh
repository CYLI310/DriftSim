#!/usr/bin/env bash
# Set up DriftSim training on a Jetson Orin Nano (JetPack 6.x, Python 3.10).
#   bash scripts/jetson/setup_jetson.sh            # venv in ~/driftsim-venv
#   VENV=/path/to/venv TORCH_INDEX=... bash scripts/jetson/setup_jetson.sh
# NVIDIA's PyTorch wheels for JetPack 6 come from the Jetson AI Lab index (jp6/cu126); a generic
# torch from pypi.org installs but has no Jetson GPU support.
set -euo pipefail
cd "$(dirname "$0")/../.."
VENV="${VENV:-$HOME/driftsim-venv}"
TORCH_INDEX="${TORCH_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu126}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"

if [ -f /etc/nv_tegra_release ]; then
  echo "L4T: $(head -1 /etc/nv_tegra_release)"
else
  echo "warning: this does not look like a Jetson (/etc/nv_tegra_release missing)"
fi
python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ needed"'

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libopenblas-dev
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip
pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX"
pip install -e ".[export]"

python - <<'PY'
import torch
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| CUDA available:", ok, "|", torch.cuda.get_device_name(0) if ok else "no GPU")
if not ok:
    raise SystemExit("CUDA is not available: remove every torch (pip uninstall torch) and rerun this script")
PY
python -c "from rc_drift_sim.rl import DriftBatchEnv; e = DriftBatchEnv(256, device='cuda'); print('DriftSim on CUDA: ok,', e.n_obs, 'observations')"

cat <<MSG

Ready. Activate with:  source "$VENV/bin/activate"
Fastest power mode:    sudo nvpmodel -q   (pick the MAXN mode)   and   sudo jetson_clocks
Quick speed check:     driftsim-train --preset safe-adaptive --device cuda --steps 2e6 --out rl_runs/speedtest
Full training:         driftsim-train --preset safe-adaptive --device cuda --out rl_runs/final --export
GUI from your laptop:  driftsim-gui --no-browser   then on the laptop  ssh -L 8765:127.0.0.1:8765 $USER@<jetson>
MSG
