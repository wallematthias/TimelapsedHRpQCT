from __future__ import annotations

from pathlib import Path

from timelapsedhrpqct.config.loader import load_config


def test_example_configs_exist_and_load() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    multistack_path = repo_root / "configs" / "example_multistack.yml"
    single_stack_path = repo_root / "configs" / "example_single_stack.yml"

    assert multistack_path.is_file()
    assert single_stack_path.is_file()

    multistack_cfg = load_config(multistack_path)
    single_stack_cfg = load_config(single_stack_path)

    assert multistack_cfg.multistack_correction.enabled is True
    assert multistack_cfg.fusion.enable_filling is True
    assert multistack_cfg.analysis.use_filled_images is True

    assert single_stack_cfg.multistack_correction.enabled is False
    assert single_stack_cfg.fusion.enable_filling is False
    assert single_stack_cfg.analysis.use_filled_images is False