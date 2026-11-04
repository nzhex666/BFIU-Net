from importlib import import_module
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
main = import_module("BFIU-Net.training").main


if __name__ == "__main__":
    raise SystemExit(main())
