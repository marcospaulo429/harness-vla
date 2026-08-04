from pathlib import Path

import pytest
from PIL import Image

from embodiedbench.evaluator.run_artifacts import create_episode_gifs, create_run_root


def test_create_run_root_refuses_to_overwrite_test(tmp_path):
    run_root = create_run_root(tmp_path, "smoke_20260804_120000")

    assert run_root == (tmp_path / "smoke_20260804_120000").resolve()
    assert (run_root / "videos").is_dir()
    with pytest.raises(FileExistsError):
        create_run_root(tmp_path, "smoke_20260804_120000")


def test_create_episode_gifs_keeps_video_inside_test_folder(tmp_path):
    run_root = create_run_root(tmp_path, "smoke")
    log_path = run_root / "base"
    image_dir = log_path / "images" / "episode_1"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4), "red").save(
        image_dir / "episode_1_step_1_front_rgb.png"
    )
    Image.new("RGB", (4, 4), "blue").save(
        image_dir / "episode_1_step_2_front_rgb.png"
    )

    gif_paths = create_episode_gifs(log_path, run_root)

    assert gif_paths == [run_root / "videos" / "episode_1.gif"]
    assert gif_paths[0].is_file()
    assert Image.open(gif_paths[0]).n_frames == 2
