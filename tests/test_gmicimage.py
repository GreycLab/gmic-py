from typing import Any, List

import gmic
import numpy as np
import pytest
from numpy.testing import assert_array_equal


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
    return np.arange(np.prod(npshape), dtype=np.float32).reshape(npshape)


@pytest.fixture
def npdata2d() -> np.ndarray:
    npshape = (2, 3, 1, 4)
    return np.arange(np.prod(npshape), dtype=np.float32).reshape(npshape)


@pytest.fixture
def img(npdata) -> gmic.Image:
    return gmic.Image(npdata.copy())


@pytest.fixture
def img2d(npdata2d) -> gmic.Image:
    return gmic.Image(npdata2d.copy())


def test_numpy_passthrough(npdata: np.ndarray, img: gmic.Image):
    assert img.shape == npdata.shape
    imgdata = img.as_numpy()
    assert isinstance(imgdata, np.ndarray)
    assert npdata.shape == imgdata.shape
    assert_array_equal(npdata, imgdata)

    imgdata = img.to_numpy()
    assert isinstance(imgdata, np.ndarray)
    assert npdata.shape == imgdata.shape
    assert_array_equal(npdata, imgdata)


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
    arr = np.asarray(mask, copy=True)
    assert_array_equal(npdata, arr)
    arr *= 0
    assert_array_equal(npdata, mask)
    arr = np.asarray(mask, copy=False)
    assert_array_equal(npdata, arr)
    arr *= 0
    assert_array_equal(img, np.zeros_like(arr))


def test_dlpack_interface(npdata: np.ndarray, img: gmic.Image):
    assert "__dlpack__" in dir(img)
    assert "__dlpack_device__" in dir(img)
    assert type(img.__dlpack__(dl_device=(1, 0))).__name__ == "PyCapsule"
    # noinspection PyTypeChecker
    arr = np.from_dlpack(img, copy=False)
    assert_array_equal(npdata, arr)
    with pytest.raises(ValueError):
        img.__dlpack__(dl_device=(2, 3))
    with pytest.raises(ValueError):
        img.__dlpack__(stream=1)


def test_at_pixel(img: gmic.Image, img2d: gmic.Image):
    arr = img.as_numpy()
    arr2d = img2d.as_numpy()
    for x in [0, 1, img.width // 2, -1, -img.width // 2]:
        for y in [0, 1, img.height // 2, -1, -img.height // 2]:
            pixel = img2d.at(x, y)
            assert len(pixel) == img2d.spectrum
            assert_array_equal(arr2d[x, y, 0], pixel)
            for z in [0, 1, img.depth // 2, -1, -img.depth // 2]:
                pixel = img.at(x, y, z)
                assert len(pixel) == img.spectrum
                assert_array_equal(arr[x, y, z], pixel)
    assert img.depth > 1
    with pytest.raises(ValueError):
        img.at(0, 0)


def test_operators(npdata: np.ndarray):
    npdata2 = npdata[::-1, ::-1, ::-1, ::-1].copy()
    i = 5
    f = 2.3

    img = gmic.Image(npdata)
    img2 = gmic.Image(npdata2)

    cimg = +img
    assert_array_equal(img, cimg, "Cloned image should be equal")
    assert img is not cimg, "Cloned image should not be same object"
    assert img.__array_interface__['data'][0] != img2.__array_interface__['data'][0], \
        "Cloned image data should not be at the same location"
    assert np.any(npdata != npdata2)
    assert img != img2, "Image should be different"

    imgorig = +img
    imgorig2 = +img2

    for fnc in ['add', 'sub', 'mul', 'truediv']:
        fname = '__{}__'.format(fnc)
        imgfnc = getattr(gmic.Image, fname)
        npfnc = getattr(np.ndarray, fname)
        ifname = '__i{}__'.format(fnc)
        imgifnc = getattr(gmic.Image, ifname)

        for op in [None, i, f, img2]:
            if op is None:
                op = 0 if fnc in ['add', 'sub'] else 1
                assert imgfnc(img, op) == img
            if op is img2 and fnc not in ['add', 'sub']:
                continue
            assert_array_equal(imgfnc(img, op), gmic.Image(npfnc(npdata, op)), "Operator should act the same as numpy")
            imgc = +img
            imgifnc(imgc, op)
            assert_array_equal(imgc, gmic.Image(npfnc(npdata, op)), "Assign-operator should act the same as numpy")
        assert_array_equal(img, imgorig, "Image should not have been modified")
        assert_array_equal(img2, imgorig2, "Image 2 should not have been modified")
