#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_task_configs.sh
#
# Copies the modified task configs from this repo into your local robot_lab
# installation. Run this once after cloning, and again after any config change.
#
# Usage:
#   ./setup_task_configs.sh                         # uses default path
#   ROBOT_LAB_DIR=/path/to/robot_lab ./setup_task_configs.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

ROBOT_LAB_DIR="${ROBOT_LAB_DIR:-$HOME/robotics/robot_lab}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[INFO] robot_lab root : $ROBOT_LAB_DIR"
echo "[INFO] repo root      : $REPO_DIR"

GO2W_DST="$ROBOT_LAB_DIR/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w"
A1_DST="$ROBOT_LAB_DIR/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1"

if [ ! -d "$GO2W_DST" ]; then
    echo "[ERROR] Go2W config directory not found: $GO2W_DST"
    echo "        Is ROBOT_LAB_DIR set correctly?"
    exit 1
fi

if [ ! -d "$A1_DST" ]; then
    echo "[ERROR] A1 config directory not found: $A1_DST"
    echo "        Is ROBOT_LAB_DIR set correctly?"
    exit 1
fi

echo "[INFO] Copying Go2W configs..."
cp "$REPO_DIR/task_configs/unitree_go2w/__init__.py"      "$GO2W_DST/__init__.py"
cp "$REPO_DIR/task_configs/unitree_go2w/rough_env_cfg.py" "$GO2W_DST/rough_env_cfg.py"
cp "$REPO_DIR/task_configs/unitree_go2w/flat_env_cfg.py"  "$GO2W_DST/flat_env_cfg.py"
cp "$REPO_DIR/task_configs/unitree_go2w/agents/sb3_sac_cfg.yaml" "$GO2W_DST/agents/sb3_sac_cfg.yaml"
cp "$REPO_DIR/task_configs/unitree_go2w/agents/sb3_ppo_cfg.yaml" "$GO2W_DST/agents/sb3_ppo_cfg.yaml"

echo "[INFO] Copying A1 configs..."
cp "$REPO_DIR/task_configs/unitree_a1/rough_env_cfg.py" "$A1_DST/rough_env_cfg.py"
cp "$REPO_DIR/task_configs/unitree_a1/flat_env_cfg.py"  "$A1_DST/flat_env_cfg.py"

echo "[INFO] Installing sbx GSAC patch..."
SBX_DIR=$(python3 -c "import sbx, os; print(os.path.dirname(sbx.__file__))" 2>/dev/null)
if [ -z "$SBX_DIR" ]; then
    echo "[WARNING] Could not locate sbx package — skipping GSAC patch."
    echo "          Activate your IsaacLab Python environment first, then re-run."
else
    cp -r "$REPO_DIR/launchers/sbx_source/gsac" "$SBX_DIR/"
    echo "[INFO] GSAC patch installed to: $SBX_DIR/gsac"
fi

echo "[OK] Setup complete."
