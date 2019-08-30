# Computer Vision Toolbox (cvToolbox)

This project created with the intention to be used mostly for educational reasons. In its final form, the toolbox will include modules for: 

* camera calibration, 
* key points detection, description & matching,
* fundamental & essential matrix calculation,
* stereo pair creation from sequential images and depth map estimation,
* 3d scene reconstruction from two images
* and - hopefully - structure from motion.

Currently, <u>only the camera calibration module has been fully developed</u>, but the key points detection, description and matching are coming soon.

## Getting Started

### Dependencies

The dependencies of this project are: 

- openCV
- numpy
- scipy
- Logger

If you use Anaconda or Miniconda, you can install the aforementioned dependencies by using the following commands (*Recommended*):

> $ conda install numpy
>
> $ conda install scipy
>
> $ conda install -c conda-forge opencv

or alternatively you can install them using `pip`.

> $ pip install numpy
>
> $ pip install scipy
>
> $ pip install opencv-python

For the installation of the *Logger* package please read the corresponding [README.md](https://github.com/ekarakasis/Logger/blob/master/README.md)


### Installation

To install this package just download this repository from GitHub or by using the following command line:

> $ git clone https://github.com/ekarakasis/cvToolbox

Afterwards, go to the local root folder, open a command line and run:

> $ pip install .

and if you want to install it to a specific Anaconda environment then write:

> $ activate <Some_Environment_Name>
>
> $ pip install .

### How to Uninstall the package

To uninstall the package just open a command and write:

> $ pip unistall cvToolbox

To uninstall it from a specific conda environment write:

> $ activate <Some_Environment_Name>
>
> $ pip unistall cvToolbox

### Run the Tests

To run the tests just go to the *root/cvToolbox/tests* folder, open a command line and write:

> $ python test_all.py


## Examples

At this point it should be mentioned that the images that have been used in the calibration process have been taken from the openCV library ([sample Images](https://github.com/opencv/opencv/tree/master/samples/data)).

### Example - Camera Calibration

```python
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import cv2
import pprint 
pp = pprint.PrettyPrinter(indent=4)
import matplotlib.pyplot as plt      
from cvToolbox.CaLib.CameraCalibration import CameraCalibration

CalibImagePath = '../../data/calibData/'
CalibImageType = '.jpg'
Image4Undistort = '../../data/calibData/left02.jpg'

# Define a class instance
cc = CameraCalibration()
# if the image names have been successfully loaded ...
if cc.LoadImageNames(CalibImagePath, typeFilter=[CalibImageType]):
    # ... then, if enough chessboards have been detected ...
    if cc.FindChessboard(
        chessboardSize=(7, 6), 
        squareSize=30, 
        showImagesFlag=False
    ):    
        # ... calibrate the camera and do something with the
        # camera matrix and the distortion coefficients.
        if cc.Calibration():
            cameraMat, distCoeffs = cc.GetCalibrationParams()
            cc.SaveParameters(fileName='intrinsicParams', save2MatFileFlag=True)
            # the parameters can also be loaded
            # cameraMat, distCoeffs = cc.LoadParameters('intrinsicParams.mat')
            
            if cameraMat is not None:
                print('\n===== Intrinsic Camera Matrix ===================')
                pp.pprint(cameraMat)
                print('')

                print('\n===== Distortion Coefficients ===================')
                pp.pprint(distCoeffs)
                print('')
                
            # undistort an image and show the result in comparison with the original one
            img = cv2.imread(Image4Undistort) 
            img_ud = cc.Undistort(img)
            
            if img_ud is not None:
                plt.figure(figsize=(20, 8))
                plt.subplot('121')
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.subplot('122')
                plt.imshow(cv2.cvtColor(img_ud, cv2.COLOR_BGR2RGB))
                plt.show()               
```

### Example - Image Features

```python
import cv2 as cv
from cvToolbox.imgFeat import ImgFeatures

# ===== create a class instance =====
# the detector, descriptor & matcher are set at this point
IF = ImgFeatures(
    detectorMethod=['fast'], 
    descriptorMethod=['orb'], 
    matchingMethod='bfmatcher'
)

# ===== read some images =====
img1 = cv.imread('../data/calibData/left02.jpg')

# ===== Detector =====
# find key points using the FAST detector
kp1 = IF.Detect(img1, ravelOutputFlag=True, showKpFlag=True)
```



## License

This project is licensed under the MIT License.
