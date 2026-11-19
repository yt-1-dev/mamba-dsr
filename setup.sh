#!/usr/bin/env bash
# Install Python 3.12 (via uv, if not already present), create a virtualenv
# for MambaDSR with it, and install requirements.txt into it.
#
#   ./setup.sh                          # clean venv in ./.venv
#   ./setup.sh --system-site-packages   # reuse the host's torch/CUDA stack
#   ./setup.sh --venv /path/to/env      # somewhere else
#   ./setup.sh --no-kernel              # skip Jupyter kernel registration
#
# On a Lightning Studio the `cloudspace` conda env already has a CUDA torch.
# A clean venv ignores it and pulls ~3 GB of wheels; --system-site-packages
# reuses what's there. Pick deliberately.

set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv"
SYSTEM_SITE=0
KERNEL=1
KERNEL_NAME="mambadsr"

while [ $# -gt 0 ]; do
    case "$1" in
        --venv)                  VENV="$2"; shift 2 ;;
        --system-site-packages)  SYSTEM_SITE=1; shift ;;
        --no-kernel)             KERNEL=0; shift ;;
        --kernel-name)           KERNEL_NAME="$2"; shift 2 ;;
        -h|--help)               sed -n '2,13p' "$0"; exit 0 ;;
        *)                       echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [ ! -f requirements.txt ]; then
    echo "error: requirements.txt not found in $(pwd)" >&2
    exit 1
fi

# ------------------------------------------------------------ python 3.12
# requirements.txt pins cp312 wheels, so get a 3.12 first. uv installs one
# into ~/.local/share/uv without root or apt/deadsnakes.
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if command -v python3.12 >/dev/null; then
        PY="python3.12"
    else
        if ! command -v uv >/dev/null; then
            echo "==> installing uv (to fetch python 3.12)"
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.local/bin:$PATH"
        fi
        echo "==> installing python 3.12"
        uv python install 3.12
        PY="$(uv python find 3.12)"
    fi
fi
command -v "$PY" >/dev/null || { echo "error: $PY not found" >&2; exit 1; }
echo "==> using $PY"

# requirements.txt pins cp312 wheels for mamba-ssm and causal-conv1d
PYVER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "$PYVER" != "3.12" ]; then
    echo "warning: python $PYVER, but requirements.txt pins cp312 wheels for"
    echo "         mamba-ssm and causal-conv1d. Those two lines need swapping"
    echo "         for your version, or use PYTHON=python3.12 ./setup.sh"
fi

# ------------------------------------------------------------ venv + pip
# Debian/Ubuntu split venv and ensurepip out of the python package, so a
# system python3.12 often can't create a venv. uv's python ships both.
if ! "$PY" -c 'import venv, ensurepip' 2>/dev/null; then
    echo "==> installing venv + pip for python $PYVER"
    SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
    if command -v apt-get >/dev/null; then
        $SUDO apt-get update -qq
        $SUDO apt-get install -y "python$PYVER-venv" python3-pip
    elif command -v dnf >/dev/null; then
        $SUDO dnf install -y "python$PYVER-pip"
    else
        echo "error: $PY has no venv/ensurepip and no apt-get/dnf to install them." >&2
        echo "       unset PYTHON and remove python3.12 from PATH to use uv's python instead." >&2
        exit 1
    fi
    "$PY" -c 'import venv, ensurepip' \
        || { echo "error: venv/ensurepip still missing for $PY" >&2; exit 1; }
fi

# ---------------------------------------------------------------- ffmpeg
# torchcodec (torchaudio.load) dlopens the system FFmpeg 4-8 shared libs.
if ! command -v ffmpeg >/dev/null; then
    echo "==> installing ffmpeg"
    SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
    if command -v apt-get >/dev/null; then
        $SUDO apt-get update -qq && $SUDO apt-get install -y ffmpeg
    elif command -v dnf >/dev/null; then
        $SUDO dnf install -y ffmpeg-free
    else
        echo "warning: ffmpeg missing and no apt-get/dnf; torchaudio.load will fail" >&2
    fi
fi

# ---------------------------------------------------------------- create venv
if [ -d "$VENV" ]; then
    VENV_VER="$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo broken)"
    if [ "$VENV_VER" != "$PYVER" ]; then
        echo "error: existing $VENV is python $VENV_VER, not $PYVER." >&2
        echo "       remove it (rm -rf $VENV) and re-run ./setup.sh" >&2
        exit 1
    fi
    echo "==> reusing existing venv at $VENV"
else
    echo "==> creating venv at $VENV"
    if [ "$SYSTEM_SITE" -eq 1 ]; then
        "$PY" -m venv --system-site-packages "$VENV"
    else
        "$PY" -m venv "$VENV"
    fi
fi

VPY="$VENV/bin/python"
[ -x "$VPY" ] || { echo "error: $VPY missing — venv creation failed" >&2; exit 1; }

# A venv left behind by an earlier run where ensurepip was missing (or made
# with --without-pip) has no pip. Bootstrap it in place rather than failing.
if ! "$VPY" -m pip --version >/dev/null 2>&1; then
    echo "==> venv has no pip, bootstrapping it"
    "$VPY" -m ensurepip --upgrade 2>/dev/null \
        || curl -sS https://bootstrap.pypa.io/get-pip.py | "$VPY"
    "$VPY" -m pip --version >/dev/null \
        || { echo "error: could not install pip into $VENV" >&2; exit 1; }
fi

# ------------------------------------------------------------------- install
echo "==> upgrading pip"
"$VPY" -m pip install --upgrade pip setuptools wheel --quiet

# Drop whatever torch stack the venv already has (older CUDA builds, or a
# mismatched torchvision/torchcodec) so requirements.txt installs the cu130 set
# fresh. mamba-ssm / causal-conv1d are compiled against torch, so they go too.
# Only touches the venv: with --system-site-packages, pip skips host packages.
echo "==> removing existing torch packages"
"$VPY" -m pip uninstall -y torch torchvision torchaudio torchcodec \
    mamba-ssm causal-conv1d 2>&1 | grep -v "not installed" || true

echo "==> installing requirements.txt (torch cu130)"
# The prebuilt mamba-ssm / causal-conv1d wheel URLs in requirements.txt need no
# build isolation flags; they are binaries, not sdists.
"$VPY" -m pip install -r requirements.txt

if [ "$KERNEL" -eq 1 ]; then
    echo "==> registering Jupyter kernel '$KERNEL_NAME'"
    # Without this the notebooks keep running on whatever interpreter they were
    # already attached to, and the venv does nothing.
    "$VPY" -m pip install ipykernel --quiet
    "$VPY" -m ipykernel install --user \
        --name "$KERNEL_NAME" --display-name "Python (MambaDSR)"
fi

# -------------------------------------------------------------------- verify
echo "==> verifying"
VERIFY_OK=0
"$VPY" - <<'PY' || VERIFY_OK=$?
import importlib, importlib.util, sys

ok = True
try:
    import torch
    print(f"  torch            {torch.__version__}")
    print(f"  cuda available   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device           {torch.cuda.get_device_name(0)}")
        print(f"  cuda (torch)     {torch.version.cuda}")
    else:
        print("  WARNING: no CUDA device visible — training will fall back to CPU")
except Exception as e:
    ok = False
    print(f"  torch            FAILED: {e}")

for mod, label in [("torchaudio", "torchaudio"), ("torchvision", "torchvision"), ("torchcodec", "torchcodec"),
                   ("mamba_ssm", "mamba-ssm"), ("causal_conv1d", "causal-conv1d"),
                   ("speechbrain", "speechbrain"), ("soundfile", "soundfile")]:
    try:
        m = importlib.import_module(mod)
        print(f"  {label:16} {getattr(m, '__version__', 'ok')}")
    except Exception as e:
        ok = False
        print(f"  {label:16} FAILED: {type(e).__name__}: {e}")

# torchaudio.load goes through torchcodec; this is the exact path that broke before
if importlib.util.find_spec("torchcodec") is not None:
    try:
        from torchcodec.decoders import AudioDecoder  # noqa: F401
        print("  torchcodec audio decoder importable")
    except Exception as e:
        ok = False
        print(f"  torchcodec audio decoder FAILED: {type(e).__name__}: {e}")
        print("    torchcodec is installed but won't load against this torch --")
        print("    that's the version mismatch; check the pins in requirements.txt")

sys.exit(0 if ok else 1)
PY

echo
if [ "$VERIFY_OK" -ne 0 ]; then
    echo "SETUP INCOMPLETE — some packages above failed to import."
    echo "The venv exists at $VENV but is not usable for training yet."
    echo "Fix the failing pins in requirements.txt and re-run ./setup.sh."
    exit 1
fi

echo "done. activate with:"
echo "    source $VENV/bin/activate"
if [ "$KERNEL" -eq 1 ]; then
    echo "or pick the 'Python (MambaDSR)' kernel in the notebook."
fi
