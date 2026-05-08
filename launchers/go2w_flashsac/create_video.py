"""Render an MP4 of a FlashSAC checkpoint without touching the user's display.

Spins up an Xvfb virtual X server, runs play.py against it, captures the
display with ffmpeg, then cleans everything up. The video lands in
<checkpoint>/videos/playback.mp4 (folder created if missing).

Usage:
    python create_video.py <checkpoint_dir>
    python create_video.py <checkpoint_dir> --duration 120 --num_episodes 10
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PLAY_SCRIPT = _SCRIPT_DIR / "play.py"


def _check_deps() -> None:
    """Fail fast with a useful message if Xvfb or ffmpeg aren't installed."""
    missing = [tool for tool in ("Xvfb", "ffmpeg") if shutil.which(tool) is None]
    if missing:
        sys.exit(
            f"Missing system tools: {', '.join(missing)}.\n"
            "Install with:  sudo apt install xvfb ffmpeg"
        )


def _kill_group(proc: subprocess.Popen, timeout: float = 3.0) -> None:
    """SIGKILL a process and any children; wait until they're gone."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("checkpoint", type=str,
                        help="Path to a checkpoint directory containing actor.pt.")
    parser.add_argument("--duration", type=int, default=60,
                        help="Recording length in seconds (default: 60).")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Number of robots to spawn (default: 1). With >1, the "
                             "viewport camera cycles between them — see --camera_switch.")
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="Episodes for play.py to roll out. If unset, auto-sized "
                             "from --duration and --num_envs so play.py outlives ffmpeg.")
    parser.add_argument("--camera_switch", type=float, default=None,
                        help="Seconds between camera cycles when --num_envs>1. "
                             "Default: --duration / --num_envs (each robot gets equal "
                             "screen time). Set 0 to keep the camera on env 0.")
    parser.add_argument("--resolution", type=str, default="1920x1080",
                        help="Xvfb resolution as WIDTHxHEIGHT (default: 1920x1080).")
    parser.add_argument("--fps", type=int, default=30,
                        help="Output video frame rate (default: 30).")
    parser.add_argument("--display", type=str, default=":99",
                        help="Xvfb display number (default: :99).")
    parser.add_argument("--warmup", type=int, default=90,
                        help="Seconds to wait for Isaac Sim to load before recording "
                             "starts (default: 90; bump if the first frames are black).")
    parser.add_argument("--output_name", type=str, default="playback.mp4",
                        help="Filename inside <checkpoint>/videos/ (default: playback.mp4).")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device for play.py (default: cuda:0).")
    args = parser.parse_args()

    _check_deps()
    if not _PLAY_SCRIPT.exists():
        sys.exit(f"play.py not found next to this script: {_PLAY_SCRIPT}")

    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not ckpt.is_dir():
        sys.exit(f"Checkpoint directory does not exist: {ckpt}")
    if not (ckpt / "actor.pt").exists():
        sys.exit(f"Missing actor.pt under: {ckpt}")

    videos_dir = ckpt / "videos"
    videos_dir.mkdir(exist_ok=True)
    out_path = videos_dir / args.output_name
    play_log_path = videos_dir / "play.log"

    width, _, height = args.resolution.partition("x")

    # Resolve auto-defaults for --camera_switch and --num_episodes based on
    # --duration and --num_envs. Each episode is ~20 s of sim, so a single
    # env produces 1 episode per 20 s of recording. Add 2 episodes of
    # headroom so play.py outlives ffmpeg.
    if args.camera_switch is None:
        args.camera_switch = (args.duration / args.num_envs) if args.num_envs > 1 else 0.0
    if args.num_episodes is None:
        episodes_per_env = max(1, int(args.duration / 20)) + 2
        args.num_episodes = episodes_per_env * args.num_envs

    # ── 1. Xvfb ─────────────────────────────────────────────────────────
    print(f"[create_video] Starting Xvfb on {args.display} at {args.resolution}")
    xvfb = subprocess.Popen(
        ["Xvfb", args.display, "-screen", "0", f"{args.resolution}x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(2)
    if xvfb.poll() is not None:
        sys.exit(f"Xvfb failed to start. Is display {args.display} already in use?")

    # ── 2. play.py — output redirected to log so it doesn't pollute stdout ─
    print(f"[create_video] Launching play.py against {args.display} (log: {play_log_path})")
    env = os.environ.copy()
    env.update({
        "DISPLAY": args.display,
        "CUDA_VISIBLE_DEVICES": "0",
        "__VK_LAYER_NV_optimus": "NVIDIA_only",
    })
    play_log = play_log_path.open("w")
    # `-u` and PYTHONUNBUFFERED=1 force play.py's stdout/stderr to flush
    # line-by-line into play.log; without them the log appears to stop
    # mid-init because Python buffers when not writing to a TTY.
    env["PYTHONUNBUFFERED"] = "1"
    play = subprocess.Popen(
        [sys.executable, "-u", str(_PLAY_SCRIPT),
         "--checkpoint", str(ckpt),
         "--num_envs", str(args.num_envs),
         "--num_episodes", str(args.num_episodes),
         "--camera_switch_interval", str(args.camera_switch),
         "--device", args.device],
        env=env, stdout=play_log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    try:
        # ── 3. Warmup — Isaac Sim loads + RTX shader cache compiles ─────
        print(f"[create_video] Warmup {args.warmup}s for Isaac Sim init...")
        for _ in range(args.warmup):
            if play.poll() is not None:
                sys.exit(f"play.py exited early during warmup. Check {play_log_path}")
            time.sleep(1)

        # ── 4. ffmpeg — captures Xvfb display to MP4 ────────────────────
        print(f"[create_video] Recording {args.duration}s @ {args.fps} fps → {out_path}")
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "x11grab",
            "-framerate", str(args.fps),
            "-video_size", args.resolution,
            "-i", args.display,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-t", str(args.duration),
            str(out_path),
        ]
        ffmpeg = subprocess.run(ffmpeg_cmd, check=False)
        if ffmpeg.returncode != 0:
            print(f"[create_video] ffmpeg returned {ffmpeg.returncode} — partial video may still be usable.")
    finally:
        # ── 5. Cleanup — SIGKILL the process groups so no Isaac stragglers ──
        print("[create_video] Cleaning up subprocesses...")
        _kill_group(play)
        _kill_group(xvfb)
        play_log.close()

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[create_video] Done. Video: {out_path}")
        print(f"[create_video] play.py log preserved at: {play_log_path}")
        return 0
    print("[create_video] No video produced. Inspect the play.py log:")
    print(f"    less {play_log_path}")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[create_video] Interrupted — cleaning up.")
        # Best-effort: kill anything still on display :99
        subprocess.run(["pkill", "-9", "-f", "Xvfb :99"], check=False)
        sys.exit(130)
