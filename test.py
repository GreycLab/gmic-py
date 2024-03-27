#!/usr/bin/env python3

from pathlib import PosixPath
try:
    import gmic
except ImportError as ex:
    import sys, os

    ex.msg += " (cwd: " + os.getcwd() + ", sys.path: " + str(sys.path) + ")"
    raise

OUT_DIR = PosixPath(__file__).parent.resolve() / "tests/out"

print(gmic.__spec__)
print(gmic.__version__, gmic.__build__)

inter = gmic.Gmic()
print(inter)
print(inter.run.__doc__)

cmdline = f"testimage2d 512 output {OUT_DIR}/testimage.png"
print("gmic>", cmdline)
imgs = inter.run(cmdline)
for img in imgs:
    print(img)
