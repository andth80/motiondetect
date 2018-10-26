import cv2
import numpy as np


def _gaussian(width, height, sigma):
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    d = np.sqrt(x * x + y * y)

    return np.exp(-((d ** 2) / sigma))


def _fft2d(s):
    assert len(s.shape) == 2

    f = cv2.dft(np.float32(s), flags=cv2.DFT_COMPLEX_OUTPUT)
    return np.fft.fftshift(f, axes=(0, 1))


class Stft2d:
    """Computes a short-time fourier transform (STFT) over a series of
    patches across an image.

    The patches are generated such that the first patch is centered at
    (offset[0], offset[1]). Subsequent patches are centred at (offset[0] +
    stride * i, offset[1] + stride * j). The image is padded with zeros as
    required to fit the patches.

    For each patch, a gaussian window function is applied and then the 2d
    FFT is calculated. The output is shifted so that the frequencies run
    from most negative to most positive (i.e. np.fft.fftshift is applied).
    Finally the output is converted to phase and magnitude data and returned
    as two arrays of shape (i, j, wx, wy) where i and j index the patches
    and wx and wy index the frequencies within a patch. The first array
    contains the phase and the second contains the magnitude.

    Arguments:
        w_size: the size of the patches in the window function
        sigma: parameter to control the rate of drop off of the Gaussian
            window function within the patch (default=0.1).
    """
    def __init__(self, w_size, sigma=0.1):
        self.w_size = w_size
        self.g_mask = _gaussian(w_size, w_size, sigma)

    def compute(self, s, stride, offset=(0, 0)):
        """Computes the transform

        :param s: input image - must be a 2d array
        :param stride: stride between patches
        :param offset: location of the center of the first patch
        :return: two arrays, the first containing the phase information,
        the second containing the magnitude.
        Both of shape (i, j, wx, wy).
        """
        if len(s.shape) != 2:
            raise ValueError('s must have dimension of 2')

        halfw = self.w_size // 2
        s = cv2.copyMakeBorder(s, halfw, halfw, halfw, halfw,
                               cv2.BORDER_CONSTANT, (0, 0, 0))

        f = [[_fft2d(s[x-halfw:x+halfw, y-halfw:y+halfw] * self.g_mask)
              for y in range(offset[1] + halfw, s.shape[1] - halfw, stride)]
             for x in range(offset[0] + halfw, s.shape[0] - halfw, stride)]

        f = np.array(f)
        a = np.arctan2(f[:, :, :, :, 0], f[:, :, :, :, 1])
        m = cv2.magnitude(f[:, :, :, :, 0], f[:, :, :, :, 1])

        return a, m
