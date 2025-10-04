import pytest
import torch
from pathlib import Path
from nequip.scripts.compile import main


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_compile_aotinductor_ase_target(tmp_path_factory, device):
    tmp_path = tmp_path_factory.mktemp("nequip_compiled")
    output_model_name = "mir-group__NequIP-OAM-L__0.1.nequip.pt2"
    output_path = Path(tmp_path) / output_model_name

    main(
        args=[
            "nequip.net:mir-group/NequIP-OAM-L:0.1",
            str(output_path),
            "--mode",
            "aotinductor",
            "--device",
            device,
            "--target",
            "ase",
        ]
    )

    assert output_path.exists()
    assert output_path.is_file()

