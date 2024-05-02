# This stub is there for py-build-cmake to be able to extract the version
# information before the native module is built

import re
from pathlib import Path

VER_REG = re.compile(r'^#define\s+gmic_version\s+(?P<version>\d+)\s*')
ROOT_DIR = Path(__file__).parents[2]

__version__ = None
__build__ = "_gmic.py stub"
with open(ROOT_DIR / "lib/gmic/src/gmic.h") as f:
    for l in f:
        match = VER_REG.match(l)
        if match is not None:
            ver = match.group("version")
            assert len(ver) == 3
            assert ver[0] == "3"
            __version__ = ".".join(ver)
            break
