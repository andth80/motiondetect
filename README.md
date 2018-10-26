Implementation of the motion detection algorithm described in this paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4737052/

The basic idea is to apply an FFT to convert frames to the frequency domain and then to compare changes in the phase of the frequency components to detect the presence of motion between the frames and also the direction and velocity of the motion.

See the [examples](https://github.com/andth80/motiondetect/blob/master/docs/example.ipynb) for details on how to use it.
