from pathlib import Path

import PIL.Image
import gmic
import numpy as np
import pytest
from numpy.testing import assert_array_equal

TEST_IMAGE = 'images/link_13x16_rgba.png'


@pytest.fixture
def img_path(request) -> Path:
    path = Path(request.fspath.dirname) / TEST_IMAGE
    assert path.is_file(), "Missing test file %s" % TEST_IMAGE
    return path


@pytest.fixture
def pil_img(img_path):
    return PIL.Image.open(img_path)


@pytest.fixture
def gmic_img(img_path) -> PIL.Image:
    return gmic.Image(img_path)


def test_pil_compat(pil_img: PIL.Image.Image, gmic_img: gmic.Image):
    assert gmic_img.shape == (pil_img.width, pil_img.height, 1, len(
        pil_img.getbands())), \
        "Images dimensions should match between gmic and PIL"

    assert_array_equal(pil_img, gmic_img.yxc,
                       "Images data should match between gmic and PIL")
    assert_array_equal(pil_img, PIL.Image.fromarray(gmic_img.yxc),
                       "Images should be identical through PIL.Image.fromarray")
    assert_array_equal(gmic.Image.from_yxc(np.asarray(pil_img)), gmic_img,
                       "Images should be identical through gmic.Image.from_yxc and np.asarray")
    img2 = +gmic_img
    img2.yxc = np.asarray(pil_img)
    assert_array_equal(img2.yxc, pil_img,
                       "Images should be identical through gmic.Image.assign and np.asarray")
    assert_array_equal(gmic.Image.from_yxc(pil_img), gmic_img,
                       "Images should be identical through gmic.Image.from_yxc")
    img2 = +gmic_img
    img2.yxc = pil_img
    assert_array_equal(img2.yxc, pil_img, "Images should be identical through gmic.Image.assign")

    with pytest.raises(ValueError):
        img3 = gmic.Image(2, 2, 1, 4)
        img3.yxc = pil_img

    img3.yxc.assign(pil_img, False)
    assert_array_equal(img3.yxc, pil_img, "Image should be identical through gmic.Image.assign")

    for mode in ['1', 'L', 'LA', 'RGB', 'I', 'I;16', 'F']:
        # noinspection PyBroadException
        try:
            conv = pil_img.convert(mode)
            img4 = gmic.Image.from_yxc(conv)
            assert img4.shape == (conv.width, conv.height, 1, len(
                conv.getbands())), \
                f"Images dimensions should match between gmic and PIL (mode {mode})"
        except Exception as ex:
            raise AssertionError(f"Got exception in from_yxc with PIL mode {mode}") from ex


def test_cast_dtype(gmic_img: gmic.Image):
    for typestr in ['u1', 'u2', 'u4', 'u8', 'i1', 'i2', 'i4', 'i8', 'b1', 'f4', 'f8']:
        dtype = np.dtype(typestr)
        arr = np.asarray(gmic_img.yxc[typestr])
        assert arr.dtype == dtype, "Array dtype should match typestring"


def test_cast_policy(gmic_img: gmic.Image):
    img = gmic_img * 4 - 512.
    imgdata = np.moveaxis(np.asarray(img)[:, :, 0, :], 0, 1)

    assert imgdata.shape == img.yxc.shape, "YXC Image shape should be identical to np array shape"
    assert imgdata.min() < 0. and imgdata.max() > 255., "Data should go beyond uint8 bounds"
    assert_array_equal(img.yxc[img.CLAMP], imgdata.clip(0, 255).astype(np.uint8), "Clamped data should match")

    np.asarray(img.yxc[img.NOCHECK])  # Shouldn't throw

    with pytest.raises(ValueError):
        np.asarray(img.yxc[img.THROW])


def test_with_z(gmic_img: gmic.Image):
    imgdata = np.asarray(gmic_img.yxc)
    img = gmic.Image(gmic_img.width, gmic_img.height, 2, gmic_img.spectrum, 0)

    with pytest.raises(RuntimeError):
        img.yxc = imgdata
    img.yxc[0] = imgdata
    assert_array_equal(img.yxc[0], imgdata)
    assert_array_equal(img.yxc[1], np.zeros_like(imgdata))
