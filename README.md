[![G'MIC Logo](https://gmic.eu/img/logo4.jpg)](https://gmic.eu)
[![Python Logo](https://www.python.org/static/community_logos/python-logo-master-v3-TM-flattened.png)](https://www.python.org)

####                                                                 

#### Python binding for G'MIC - A Full-Featured Open-Source Framework for Image Processing

##### https://gmic.eu

---------------------------

# gmic-py

`gmic-py` is the official Python 3 binding for the [G'MIC C++ image processing library](https://gmic.eu) purely written
with Python's C API. This project lives under the CeCILL license (similar to GNU Public License).

You can use the `gmic` Python module for projects related to desktop or server-side graphics software, numpy,
video-games, image procesing.

Note: the package has been completely reworked since version 2.x, the documentation and examples have been removed until
they're updated again. The "gmic" package on pypi has not been updated yet. The old binding can be found on branch

## Quickstart

First install the G'MIC Python module in your (virtual) environment.

```sh
git clone --recursive https://github.com/GreycLab/gmic-py
cd gmic-py
pip install .
```

G'MIC is a language processing framework, interpreter and image-processing scripting language. Here is how to load
`gmic`, and evaluate some G'MIC commands with an interpreter.

```python
import gmic

# On Linux a window shall open-up and display a blurred earth
gmic.run("sp earth blur 4 display")
# Filter a rose with bokeh effect and get the result as a gmic.ImageList
imglst = gmic.run("sp rose fx_bokeh 3,8,0,30,8,4,0.3,0.2,210,210,80,160,0.7,30,20,20,1,2,170,130,20,110,0.15,0")
# Save the image from the previous run() to a file
gmic.run("output rose_with_bokeh.png", imglst)
```

## Building

This project uses [Scikit-build-core](https://scikit-build-core.readthedocs.io/) which is a PEP-517-style build backend.
You can build it with `python -m build` but first you need to put the version string in version.txt at the root of the
repository (skbuild does not have a pre-build hook, sadly).
`./version_build.py -u` will do that, and can even invoke the build module or pip afterwards, like so :

```shell
$ ./version_build.py -u # Calculates the version string and writes it into version.txt
$ ./version_build.py -u && python -m build -v # Calculates and write the version then build sdist and wheel files
# Equivalent shortcut: 
$ ./version_build.py --build -v # Calculates and write the version then invoke python -m build -swn (anything after --build is passed on to build).
$ ./version_build.py --install -v . # Calculates and write the version then invoke python -m pip install -v . (anything after --install is passed on to pip)
```

All of the above will produce Release builds, set the DEBUG environment variable for Debug builds (e.g
`DEBUG=1 ./version_build.py --build`). Building gmic-py implies building gmic, which can take a while. For developpment
you can save a lot of time by disabling build isolation (`-n` for build or `--no-build-isolation` for pip), which will
reuse the same build directory (./build/debug or ./build/release) and thus not rebuild gmic unless necessary.

## Versioning

The version of the binding calculated by `version_build.py` is formatted `X.Y.Z[.rR][.devD]` (i.e 3.3.5.r2.dev5),
according to the following logic:

* Each update of the gmic submodule is tagged `gmic-X.Y.Z`
* Each commit on the main branch (since the beginning of this scheme) is a merge commit, its version is `X.Y.Z` for the
  first commit for a given gmic version, then `X.Y.Z.rR` with each commit incrementing R by 1.
* Each commit on the dev branch (or any other branch) gets the version that would be on the stable branch if it were
  merged, plus `.devD`, with D a number of commits (including the current one, first-parents only):
    * since the last merge, if the gmic upstream version is still the same
    * since the last gmic submodule update otherwise

To make the script usable even on older commits on dev branches, if the given ref has already been
merged into stable, then it will pretend stable to be the last stable commit before said merge
(e.g in the graph below, running version_build on 1c7d5ec will consider 'stable' to be cf1d5e6 even if 7b32186 already
exists).
Here's an example of git history with the version corresponding to each commit between brackets. Stable commits on the
left branch, dev on the right :

```
[1.2.4]          *   7b32186 (branch: main) Merged version 1.2.4 into main
                 |\
[1.2.4.dev2]     | * 1c7d5ec (branch:dev) So much stuff
[1.2.4.dev1]     | * 694a164 (tag: gmic-1.2.4) Updating gmic to 1.2.4
[1.2.3.r1]       * | cf1d5e6 Merged version 1.2.3.r1 into main
                 |\|
[1.2.3.r1.dev1]  | * b671ec6 Stuff again
[1.2.3]          * | 6bc2e3a Merged version 1.2.3 into main
                 |\|
[1.2.3.dev2]     | * 567ae3f More stuff
[1.2.3.dev1]     | * 03d0ea9 (tag: gmic-1.2.3) Updated gmic to 1.2.3
```