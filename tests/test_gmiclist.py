import gmic
import numpy as np
import numpy.testing as nptest
import pytest


@pytest.fixture
def npdata():
    npshape = (2, 3, 4, 5)
    return np.arange(np.prod(npshape)).reshape(npshape)


@pytest.fixture
def img(npdata):
    return gmic.Image(npdata.copy())


def test_construct(img):
    lst = gmic.ImageList()
    assert len(lst) == 0

    lst = gmic.ImageList([])
    assert len(lst) == 0

    lst = gmic.ImageList([img])
    assert len(lst) == 1
    nptest.assert_array_equal(img, lst[0])
