import cv2
import numpy as np


class Radon:
    """Simple and fast Radon transform implementation

    This is implemented as a class in order to pre-compute and cache the
    rotation matrices, which enables repeated transforms with the same
    parameters to be performed more efficiently.

    Arguments:
        radius: the radius of the input images must be square and single
            channel - i.e. arrays of shape (radius*2, radius*2)
        thetas: a list of angles for which the transform will be computed;
            the output of the transform will be of shape
            (radius*2, len(thetas))
    """
    def __init__(self, radius, thetas):
        self.diameter = radius * 2
        self.rotations = [cv2.getRotationMatrix2D((radius, radius), -theta, 1)
                          for theta in thetas]

    def compute(self, image):
        """Applies the transform to the given input

        :param image: input image, must be an array of shape
        (2*radius, 2*radius)
        :return: transformed image
        """
        if (image.shape[0] != self.diameter
                or image.shape[1] != self.diameter
                or len(image.shape) != 2):
            raise ValueError(
                'image must have size [{0}, {0}]'.format(self.diameter))

        ret = np.empty((self.diameter, len(self.rotations)))

        for i, rotation in enumerate(self.rotations):
            rotated = cv2.warpAffine(image, rotation,
                                     (self.diameter, self.diameter),
                                     cv2.INTER_NEAREST)
            ret[:, i] = np.sum(rotated, axis=0)

        return ret

