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

## Quickstart

First install the G'MIC Python module in your (virtual) environment.

```sh
git clone --recursive -b nanobind https://github.com/GreycLab/gmic-py
cd gmic-py
pip install .
```

G'MIC is a language processing framework, interpreter and image-processing scripting language. Here is how to load
`gmic`, and evaluate some G'MIC commands with an interpreter.

```python
import gmic

gmic.run("sp earth blur 4 display")  # On Linux a window shall open-up and display a blurred earth
gmic.run(
    "sp rose fx_bokeh 3,8,0,30,8,4,0.3,0.2,210,210,80,160,0.7,30,20,20,1,2,170,130,20,110,0.15,0 output rose_with_bokeh.png")  # Save a rose with bokeh effect to file
```

