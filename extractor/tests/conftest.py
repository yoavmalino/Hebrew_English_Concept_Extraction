import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = Path(__file__).parent / "golden"


def tesseract_has_hebrew() -> bool:
    exe = shutil.which("tesseract")
    if not exe:
        return False
    out = subprocess.run([exe, "--list-langs"], capture_output=True, text=True)
    return "heb" in out.stdout.split()


@pytest.fixture(scope="session")
def fixtures() -> Path:
    return FIXTURES
