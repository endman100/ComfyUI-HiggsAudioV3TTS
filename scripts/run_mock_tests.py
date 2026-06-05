from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tests.test_higgs_audio_v3_tts as tests


def main() -> int:
    for name in sorted(n for n in dir(tests) if n.startswith("test_")):
        getattr(tests, name)()
        print(f"{name} ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
