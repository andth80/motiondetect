Implementation of the motion detection algorithm described in this paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4737052/

The basic idea is to apply an FFT to convert frames to the frequency domain and then to compare changes in the phase of the frequency components to detect the presence of motion between the frames and also the direction and velocity of the motion.

See the [examples](https://github.com/andth80/motiondetect/blob/master/docs/example.ipynb) for details on how to use it.

# Install

1. install dependencies

* Python 3

2. install package

```
git clone https://github.com/andth80/motiondetect.git
cd motiondetect
pip install .
```

# To run the examples

1. install jupyter and pillow

```
pip install jupyter
pip install pillow
```

2. open the examples in the docs directory
* `jupyter notebook`
* and browse to the docs directory
