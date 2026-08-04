"""Organize local evaluation artifacts without changing benchmark behavior."""

from pathlib import Path
import re

from PIL import Image


_TEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def create_run_root(runs_root, test_id):
    """Create one immutable directory for a single evaluation attempt."""
    if not isinstance(test_id, str) or not _TEST_ID_PATTERN.fullmatch(test_id):
        raise ValueError("test_id must use only letters, numbers, '.', '_' or '-'")
    run_root = Path(runs_root).expanduser().resolve() / test_id
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "videos").mkdir()
    return run_root


def create_episode_gifs(log_path, run_root, duration_ms=700):
    """Render front-camera frames into one reviewable GIF per episode."""
    image_root = Path(log_path) / "images"
    video_root = Path(run_root) / "videos"
    video_root.mkdir(parents=True, exist_ok=True)
    gif_paths = []

    for episode_dir in sorted(image_root.glob("episode_*")):
        image_paths = sorted(
            episode_dir.glob("*_front_rgb.png"),
            key=lambda path: int(path.name.split("_step_")[1].split("_")[0]),
        )
        frames = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                frames.append(image.convert("RGB"))
        if not frames:
            continue
        gif_path = video_root / f"{episode_dir.name}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
        gif_paths.append(gif_path)

    return gif_paths
