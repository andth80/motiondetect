from distutils.core import setup

setup(
    name='MotionDetect',
    version='0.1.0',
    author='Andrew Thomas',
    author_email='athomas@fastmail.co.uk',
    packages=['motiondetect', 'motiondetect.transform'],
    url='',
    license='LICENSE',
    description='Simple phase based motion detection.',
    install_requires=[
        "numpy >= 1.14.3",
        "opencv-python >= 3.4.2.17",
    ],
)
