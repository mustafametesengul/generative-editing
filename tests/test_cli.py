import pytest
from PIL import Image

from generative_editing.cli import Cli
from generative_editing.decomposition import Weather


def test_cli_does_not_write_candidate_when_all_gates_fail(tmp_path) -> None:
    source_path = tmp_path / "black.png"
    output_path = tmp_path / "output.png"
    Image.new("RGB", (64, 64), "black").save(source_path)
    cli = Cli.model_construct(
        input=source_path,
        output=output_path,
        weather=Weather.CLEAR,
        editor="mock",
        decomposer="heuristic",
        seed=0,
        candidates=1,
        cpu_offload=False,
        debug_dir=None,
    )

    with pytest.raises(SystemExit, match="No candidate passed"):
        cli.cli_cmd()

    assert not output_path.exists()
