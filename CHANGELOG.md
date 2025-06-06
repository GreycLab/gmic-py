# Changelog

`gmic-py` embeds a libgmic C++ recompiled .so and adds C/Python binding to it.
Here are major changes to the project.

All versions listed here correspond to
a [downloadable package release from G'MIC's pypi.org project](https://pypi.org/project/gmic/#history).

## 2.9.4-alpha1 (2020-12-23)

- just wrapping ligbmic 2.9.4
- replacing uses of `_document_gmic` in docs/ by `_parce_cli_images append x`
- pytest suite not green yet

## 2.9.2 (2020-12-21)

- nothing new, just wrapping ligbmic 2.9.1
- the pytest suites are still not green, this is intened and will be fixed for 2.9.4
- releasing this as a stable version which spares users from specifying a `==<version number>` suffix when pip
  installing
- almost synchronous with
  the [tutorials 1,2 & 3 end of writing](https://gmic-py.readthedocs.io/en/latest/tutorials.html).

## 2.9.1-alpha6 (2020-12-09)

- discarding libcurl as a dependency, relying on libgmic/CImg's `curl` executable
  detection https://github.com/myselfhimself/gmic-py/issues/82
- still linux only for now

## 2.9.1-alpha5 (2020-10-23)

- embedding recompiled libgmic 2.9.1
- alpha release with only toolkit-related pytest suites green
- now more fledged documentation at https://gmic-py.readthedocs.io/
- Early input/output support for Numpy, PIL, Scikit-Image.
- Linux support only, macOS support is dropped.

## 2.9.0 (2020-06-21)

- embedding recompiled libgmic 2.9.0
- binding of C++ `gmic::gmic`` class and callable class as Python `gmic.Gmic` type.
- binding of C++ `gmic\_image`` class as Python `gmic.GmicImage` type.
- C++ `gmic_list` class is discarded for binding in favor of pure-Python lists.
- MacOS and Linux support only.

## 0.0.1 (2019-12-03)

- First fiddling C/Python vs. cffi libcmig Python marshalling tests
