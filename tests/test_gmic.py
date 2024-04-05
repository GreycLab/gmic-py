import numpy as np
import numpy.testing as nptest

import gmic

npshape = (2, 3, 4, 5)
npdata = np.arange(np.prod(npshape)).reshape(npshape)


def test_numpy_passthrough():
    img = gmic.GmicImage(npdata.copy())
    assert img.shape == npdata.shape
    imgdata = img.to_ndarray()
    assert npdata.shape == imgdata.shape
    nptest.assert_array_equal(npdata, imgdata)