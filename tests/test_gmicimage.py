from typing import Any, List

import numpy as np
import numpy.testing as nptest
import pytest

import gmic


class AttrMask:
    obj: Any
    attrs: List[str]

    def __init__(self, obj: Any, *attrs: str):
        self.obj = obj
        self.attrs = list(attrs)

    def __getattr__(self, item):
        if item in self.attrs:
            return self.obj.__getattr__(item)
        raise AttributeError(obj=self.obj, name=item)


@pytest.fixture
def npdata():
    npshape = (2, 3, 4, 5)
    return np.arange(np.prod(npshape)).reshape(npshape)


@pytest.fixture
def img(npdata):
    return gmic.GmicImage(npdata.copy())


def test_numpy_passthrough(npdata: np.ndarray, img: gmic.GmicImage):
    assert img.shape == npdata.shape
    imgdata = img.to_ndarray()
    assert npdata.shape == imgdata.shape
    nptest.assert_array_equal(npdata, imgdata)


def test_numpy_resize(npdata: np.ndarray):
    for arr, shp in [(npdata[0], ),
                (npdata[0, 0]),
                (npdata[0, 0, 0])]:
        img = gmic.GmicImage(arr)


def test_array_interface(img: gmic.GmicImage):
    assert isinstance(img.__array_interface__, dict)
    mask = AttrMask(img, "__array_interface__")
    arr = img.to_ndarray()
    arr2 = np.array(mask)
    nptest.assert_array_equal(arr, arr2)


def test_dlpack_interface(img: gmic.GmicImage):
    assert isinstance(img.__dlpack__, dict)
    mask = AttrMask(img, "__array_interface__")
    arr = img.to_ndarray()
    arr2 = np.array(mask)
    nptest.assert_array_equal(arr, arr2)
