"""Check whether PyTorch can execute work on the available CUDA GPU."""

import sys

import torch


def main() -> int:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda or 'none'}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("No CUDA device is visible to PyTorch.", file=sys.stderr)
        return 1

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    total_gib = properties.total_memory / 1024**3
    print(f"Device: {properties.name}")
    print(f"Compute capability: {properties.major}.{properties.minor}")
    print(f"VRAM: {total_gib:.1f} GiB")

    left = torch.randn((2048, 2048), device=device)
    right = torch.randn((2048, 2048), device=device)
    result = left @ right
    torch.cuda.synchronize(device)

    print(f"Matrix result device: {result.device}")
    print(f"Matrix result checksum: {result.mean().item():.6f}")
    print("CUDA execution succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())