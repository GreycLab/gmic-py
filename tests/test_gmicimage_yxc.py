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
    assert pil_img.width == gmic_img.width and pil_img.height == gmic_img.height and len(
        pil_img.getbands()) == gmic_img.spectrum and gmic_img.depth == 1

    assert_array_equal(pil_img, PIL.Image.fromarray(gmic_img.yxc))
    assert_array_equal(gmic.Image.from_yxc(np.asarray(pil_img)), gmic_img)
    assert_array_equal(gmic.Image.from_yxc(pil_img), gmic_img)

    for mode in ['1', 'L', 'LA', 'RGB', 'I', 'I;16', 'F']:
        gmic.Image.from_yxc(pil_img.convert(mode))
