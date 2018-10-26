import numpy as np
import math
import cv2

from motiondetect.transform import Stft2d
from motiondetect.transform import Radon


def _circle_mask(shape):
    radius = min(shape) // 2
    c0, c1 = np.ogrid[0:shape[0], 0:shape[1]]
    mask = ((c0 - shape[0] // 2) ** 2 + (c1 - shape[1] // 2) ** 2)
    mask = mask <= radius ** 2

    return mask


def _convert_to_mono(s):
    if len(s.shape) > 2:
        s = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)
    return s


def _convert_to_rgb(s):
    if len(s.shape) == 2:
        s = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)
    return s


def _num_diff(a1, a2, m1, m2, epsilon, gamma):
    # use the magnitude to weight the derivatives as we only really
    # care about changes in the prominent frequencies
    delta_a = (a2 - a1) * np.clip(np.minimum(m1, m2), 0, gamma) / 2

    # there are some extreme values which seem to be just noise / artifacts;
    # setting these to zero seems to greatly improve the results. TODO: come
    # up with a proper theoretical justification for this step.
    mask = np.clip(delta_a, epsilon, None) + \
        np.clip(delta_a, None, -epsilon) == 0

    return delta_a * mask


def overlay(s, motion_data, stride, offset=(0, 0)):
    """Overlays arrows showing direction of motion onto an image

    :param s: the image to overlay arrows onto, can be 2d (gray-scale) or
        3d (RGB) array.
    :param motion_data: the motion data to overlay
    :param stride: the stride between data points in the motion data
    :param offset: the location on the image of the first data point in the
        motion data
    :return: an RGB image with arrows showing the direction and magnitude of
        motion
    """
    vis = np.copy(s)
    vis = _convert_to_rgb(vis)

    yellow = (255, 255, 0)

    for (i, j, d), v in np.ndenumerate(motion_data):
        if v > 0:
            x = j * stride + offset[0]
            y = i * stride + offset[1]
            r = np.deg2rad(d * 360 / motion_data.shape[2])

            start = (x, y)
            end = (x + int(v * math.cos(r) * stride / 4),
                   y + int(-v * math.sin(r) * stride / 4))

            cv2.arrowedLine(vis, start, end, yellow, tipLength=1.5/v)

    return vis


class _Detector:
    def __init__(self, w_size=32, theta_step_size=15, epsilon=1, gamma=1.0):
        self.w_size = w_size
        self.epsilon = epsilon
        self.gamma = gamma

        self.c_mask = _circle_mask((w_size, w_size))

        self.theta_step_size = theta_step_size
        self.thetas = np.linspace(0., 180., 180 // theta_step_size,
                                  endpoint=False)
        self.radon = Radon(w_size // 2, self.thetas)
        self.c = self.radon.compute(np.ones((w_size, w_size)) * self.c_mask)

    def _detect_motion_ij(self, delta_a):
        # mask the image to a circle so that when it is rotated in the radon
        # transform, none of it is clipped
        r = self.radon.compute(delta_a * self.c_mask)

        # the lines along which the radon transform performs its sums will
        # intersect with different amounts of the circle depending on how far
        # the line passes from the middle of the circle; so we scale the
        # output of the radon transform to account for this by dividing each
        # component of the output by the length of the intersection between
        # the line that generated it and the circle.
        #
        # The length of intersection is just given by the radon transform of
        # the mask image - which is 1 inside the circle and 0 outside.
        r = np.divide(r, self.c, out=np.zeros_like(r), where=self.c != 0)

        # when something is moving within the window, dp/dt(wx,wy) looks
        # like parallel lines of ~equal values oriented perpendicular to the
        # direction of motion; because some lines of values are negative and
        # some are positive we can find the direction they point in by
        # looking for the value of theta which maximises the formula below (
        # the maximum occurs when the lines in the radon transform line up
        # with the lines of equal values in the dp/dt output).
        degree_of_alignment = np.sum(np.abs(r), axis=0)
        i = np.argmax(degree_of_alignment)
        direction = i * self.theta_step_size

        # the larger this number, the closer dp/dt is to being a set of
        # parallel lines of equal value - which is characteristic of
        # the changes in phase seen when a shape is moving across the window
        motion_indicator = degree_of_alignment[i]

        # work out whether the direction is positive or negative along the
        # detected direction; we can work this out by looking at the sign of
        # the derivative for the lowest +ve frequencies (we have to ignore
        # the higher frequencies as the phase for those starts getting
        # aliased if the velocity is too high and so the sign of the
        # derivative can become reversed); in fact, by counting the sign
        # changes as the frequencies increase we can also get an approximation
        # of the velocity.
        #
        # use a 2-period moving average to smooth out any noise
        mid = r.shape[0] // 2
        signs = np.signbit(r[mid:-2, i] + r[mid + 1:-1, i])

        if signs[0]:
            direction += 180

        velocity = len(np.where(np.diff(signs))[0]) + 1

        return motion_indicator, direction, velocity

    def detect(self, a1, m1, a2, m2, threshold):
        delta_a = _num_diff(a1, a2, m1, m2, self.epsilon, self.gamma)

        ret = np.zeros((delta_a.shape[0], delta_a.shape[1],
                        360 // self.theta_step_size))

        for i in range(delta_a.shape[0]):
            for j in range(delta_a.shape[1]):
                p, d, v = self._detect_motion_ij(delta_a[i, j, :, :])

                if p > threshold:
                    ret[i, j, d // self.theta_step_size] = v

        return ret


class SequenceTracker:
    """Takes a series of images and returns data indicating motion.

    For example the frames may come from a video file. The frames are passed in
    sequentially and each frame is compared to the previous one and data
    returned about the motion detected between them.

    It creates a grid of detection points across the image and for each
    point examines a square patch/window of the image around the point to
    see if there is motion at that point.

    Arguments:
        stride: the length (in pixels) between motion detection points
        threshold: controls the sensitivity of the detector and whether it
            determines there is motion at a given point
        offset: the location of the first detection point (default=(0,0))
        w_size: the size (in pixels) of the detection window around a
            detection point (default=32)
        theta_step_size: the granularity (in degrees) to which the direction
            can be detected (default=15)
        epsilon: configuration parameter that controls noise reduction in the
            phase derivatives (default=1)
        sigma: configuration parameter that contols the rate of drop off of
            the Gaussian window function (default=0.1)
        gamma: another configuration parameter that controls noise reduction in
            the phase derivatives
    """
    def __init__(self, stride, threshold, offset=(0, 0), w_size=32,
                 theta_step_size=15, epsilon=1, sigma=0.1, gamma=1.0):

        self.stft = Stft2d(w_size, sigma)
        self.detector = _Detector(
            w_size=w_size, theta_step_size=theta_step_size, epsilon=epsilon,
            gamma=gamma)

        self.stride = stride
        self.offset = offset
        self.threshold = threshold
        self.old_a = None
        self.old_m = None

    def next_frame(self, s):
        """Detects motion between this frame and the previous one.

        Takes the passed in frame, compares it to the previous one and
        returns data on the detected motion. Then stores this frame to use
        as the comparison for the next call to the method.

        :param s: 2D (gray-scale) or 3D (channels can be in any order) array
            representing the frame to be compared
        :return: an array of form [i, j, d] where i and j index the
            detection points; and d indexes angles of direction
            i.e. [0, 1, 2, ..., 360 // theta_step_size]. The values in the
            array are zero if no motion was detected at that point and in
            that direction, or the velocity of the motion if there was motion
            detected. For a given i, j, a maximum of one element of [i, j, :]
            will be non-zero.
        """
        s = _convert_to_mono(s)
        a, m = self.stft.compute(s, stride=self.stride, offset=self.offset)

        if self.old_a is None or self.old_m is None:
            # this is the first frame
            self.old_a = a
            self.old_m = m

        d = self.detector.detect(self.old_a, self.old_m, a, m,
                                 threshold=self.threshold)

        self.old_a, self.old_m = a, m

        return d

    def overlay(self, s, motion_data):
        """Overlays arrows showing direction of motion onto an image

        :param s: the image to overlay arrows onto, can be 2d (gray-scale) or
            3d (RGB) array.
        :param motion_data: the motion data to overlay
        :return: an RGB image with arrows showing the direction and
            magnitude of motion
        """
        return overlay(s, motion_data, self.stride, self.offset)

