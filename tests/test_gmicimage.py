from typing import Any, List

import PIL.Image
import gmic
import numpy as np
import numpy.testing as nptest
import pytest


class AttrMask:
    obj: Any
    attrs: List[str]

    def __init__(self, obj: Any, *attrs: str):
        self.obj = obj
        self.attrs = list(attrs)

    def __getattr__(self, item):
        if item in self.attrs:
            return getattr(self.obj, item)

        raise AttributeError(self.obj, item)

    def __dir__(self):
        d = list(object.__dir__(self))
        d += self.attrs
        return d


@pytest.fixture
def npdata() -> np.ndarray:
    npshape = (2, 3, 4, 5)
    return np.arange(np.prod(npshape)).reshape(npshape)


@pytest.fixture
def img(npdata) -> gmic.Image:
    return gmic.Image(npdata.copy())


def pil_img() -> PIL.Image:
    npshape = (4, 5, 3)
    return PIL.Image.fromarray(np.arange(np.prod(npshape), dtype=np.uint8).reshape(npshape), "RGB")


def test_numpy_passthrough(npdata: np.ndarray, img: gmic.Image):
    assert img.shape == npdata.shape
    imgdata = img.as_numpy()
    assert isinstance(imgdata, np.ndarray)
    assert npdata.shape == imgdata.shape
    nptest.assert_array_equal(npdata, imgdata)
    
    imgdata = img.to_numpy()
    assert isinstance(imgdata, np.ndarray)
    assert npdata.shape == imgdata.shape
    nptest.assert_array_equal(npdata, imgdata)


def test_numpy_resize(npdata: np.ndarray):
    sh = npdata.shape
    for arr, shp in [(npdata[0], (sh[1], sh[2], sh[3], 1)),
                     (npdata[0, 0], (sh[2], sh[3], 1, 1)),
                     (npdata[0, 0, 0], (sh[3], 1, 1, 1))]:
        img = gmic.Image(arr)
        assert shp == img.shape


def test_array_interface(npdata: np.ndarray, img: gmic.Image):
    assert isinstance(img.__array_interface__, dict)
    assert "__array_interface__" in dir(img)
    mask = AttrMask(img, "__array_interface__")
    arr = np.array(mask)
    nptest.assert_array_equal(npdata, arr)


def test_dlpack_interface(npdata: np.ndarray, img: gmic.Image):
    assert "__dlpack__" in dir(img)
    assert "__dlpack_device__" in dir(img)
    assert type(img.__dlpack__()).__name__ == "PyCapsule"
    # noinspection PyTypeChecker
    arr = np.from_dlpack(img)
    nptest.assert_array_equal(npdata, arr)

#
# def test_operators(npdata: np.ndarray, img: gmic.Image):
#     nptest.assert_array_equal()
