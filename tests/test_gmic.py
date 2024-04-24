import numpy as np
import numpy.testing as nptest

import gmic

npshape = (2, 3, 4, 5)
npdata = np.arange(np.prod(npshape)).reshape(npshape)
img = gmic.GmicImage(npdata.copy())


def test_numpy_passthrough():
    assert img.shape == npdata.shape
    imgdata = img.to_ndarray()
    assert npdata.shape == imgdata.shape
    nptest.assert_array_equal(npdata, imgdata)


def test_array_interface():
    assert isinstance(img.__array_interface__, dict)
    arr = img.to_ndarray()
    arr2 = np.array(img)
    nptest.assert_array_equal(arr, arr2)
