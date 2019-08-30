#  ==================================================================================
#  
#  Copyright (c) 2019, Evangelos G. Karakasis 
#  
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#  
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  
#  ==================================================================================

import cv2 as _cv
import time as _time
import numpy as _np
import matplotlib.pyplot as _plt 
from Logger.Logger import Logger as _Logger   



# ===== UTILITIES ======================================================================================================

# ------------
# Kill all previous loggers and delete the corresponding log files  
# BE CAREFUL!!! this lines may kill and delete other useful loggers and 
# log files in your application.
# _Logger.KillAllLoggers()
# _Logger.DeleteAllLogFiles()
# ------------


class _ImgFeatLogger: 
    TNLM = _Logger('ImgFeatures.tnlm', 'imageFeaturesLog', 'Logs', 'time-name-level-message')
    
    @staticmethod
    def Log(logFlag, logLevel, *args):
        _ImgFeatLogger.TNLM.Log(logFlag, logLevel, *args)
        
    @staticmethod
    def KillLogger():
        _ImgFeatLogger.TNLM.CloseLogger()  

    @staticmethod
    def KillLogger_n_DeleteLogFiles():
        _ImgFeatLogger.TNLM.CloseLogger()        
        _ImgFeatLogger.TNLM.DeleteLogFile()   


class _myTimer:
    """A simple timer class, which allows measuring code duration using tic toc functions.
    
    Example
    ------- 
    myTimer.tic('myFirstTimer')
    
    for i in range(100000):
        pass

    myTimer.tic('mySecondTimer')

    for i in range(100000):
        pass

    sf = myTimer.toc('mySecondTimer')
    ff = myTimer.toc('myFirstTimer')

    print('The duration of the second for is : {}'.format(sf))
    print('The duration of both for is: {}'.format(ff))
    """
    lastTic = {}
    
    @staticmethod
    def tic(timerName):
        _myTimer.lastTic[timerName] = _time.time()
    
    @staticmethod
    def toc(timerName): # in msec
        duration = round(1000*(_time.time() - _myTimer.lastTic[timerName]), ndigits=3)
        _myTimer.lastTic.pop(timerName, None)        
        return duration
    
    
# ===== DETECTORS ======================================================================================================
# This classes HarrisDetector and  ShiTomasiDetector have been developed 
# in order to have a way of applying the corresponding detectors, in a similar 
# manner with other detectors such as ORB. Detectors like ORB, FAST etc. 
# make use of a function 'detect', which takes two parameters as input.
# Thus, the implemented classed have a static function called detect, which takes
# two parameters: the image in which we want to find the key points, and 
# a 'uselessArg' parameter, which is always None just to imitate the way
# the other detectors work. This is helpfull enough in generalizing the concept
# of Detector. I know it's not the best practice, but it works.


class _HarrisDetector:  
    """Applies the Harris detector using the opencv.     
    """
    @staticmethod
    def detect(img, uselessArg=None):  
        """Applies the Harris detector.
        
        Parameters 
        ---------- 
        img: 2d ndarray
            The image in which we want to detect the key points.
        uselessArg: fake parameter
            It is used only to imitate the way that other detectors work.
        """
        featureThress = 5000.        
        dst = _cv.cornerHarris(img, blockSize=2, ksize=3, k=0.04)
        
        # Although that Harris is an extremely fast detector
        # the particular implementation where we want to return
        # a list of keypoints makes it slow enough!
        
        dst[dst>(1./featureThress)*dst.max()] = 255  
        dst = _np.uint8(dst)
        
        # find centroids
        ret, labels, stats, centroids = _cv.connectedComponentsWithStats(dst)
        
        # define the criteria to stop and refine the corners
#         if criteriaIterations is not None:
        criteria = (_cv.TERM_CRITERIA_EPS + _cv.TERM_CRITERIA_MAX_ITER, 10, 0.001)
        centroids = _cv.cornerSubPix(img, _np.float32(centroids), (5,5), (-1,-1), criteria)
        
        kp = [_cv.KeyPoint(crd[0], crd[1], 16) for crd in centroids]
        return kp
    
class _ShiTomasiDetector:  
    """Applies the Shi-Tomasi detector using the opencv.     
    """
    @staticmethod
    def detect(img, uselessArg=None):    
        """Applies the Shi-Tomasi detector.
        
        Parameters 
        ---------- 
        img: 2d ndarray
            The image in which we want to detect the key points.
        uselessArg: fake parameter
            It is used only to imitate the way that other detectors work.
        """
        maxCorners = 5000
        corners = _cv.goodFeaturesToTrack(
            img, 
            maxCorners=maxCorners, 
            qualityLevel=0.01, 
            minDistance=3, 
            useHarrisDetector=False
        )
        kp = [_cv.KeyPoint(crd[0][0], crd[0][1], 16) for crd in corners]      
        return kp     


# ===== IMAGE FEATURES CLASS ===========================================================================================

class ImgFeatures:
    """A flexible class responsible for key point detection, description and matching.
    
    Parameters 
    ---------- 
    detectorMethod: list of strings, (optional, default: ['fast'])
        A list that includes the names of the desired detectors. The FAST detector is used
        by default. The supported detectors are:
            * 'orb',
            * 'harris',
            * 'brisk',
            * 'fast',
            * 'shitomasi',
            * 'harrislaplace'
            
        Example: IF = ImgFeatures(detectorMethod=['fast', 'orb'])
            
    descriptorMethod: list of strings, (optional, default: ['orb'])
        A list that includes the names of the desired descriptors. The ORB descriptor is used
        by default. The supported descriptors are:
            * 'org',
            * 'brisk'
        
        The SIFT and SURF descriptors and their corresponding detectors have not been included
        due to patent issues. However, it is straightforward enough to be included. It is just
        neede a proper update of the constructor (__init__ ).
        
        Example: IF = ImgFeatures(detectorMethod=['fast'], descriptorMethod=['orb'])
        
    matchingMethod: str, (optional, default: 'bfmatcher')
        A string that includes the name of the desired matcher. The BFMatcher matcher is used
        by default. The supported matchers are:
            * 'bfmatcher',
            * 'flannmatcher'
            
        Example: IF = ImgFeatures(detectorMethod=['fast'], descriptorMethod=['orb'], matchingMethod='bfmatcher')
        
        
    Examples
    -------- 
    
    # ===== create a class instance =====
    # the detector, descriptor & matcher are set at this point
    IF = ImgFeatures(detectorMethod=['fast'], descriptorMethod=['orb'], matchingMethod='bfmatcher')
        
    # ===== read some images and produce their gray version =====
    img1 = cv.imread('../data/calibData/left02.jpg')
    imgG1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    
    img2 = cv.imread('../data/calibData/left05.jpg')    
    imgG2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    
    # ===== Detector =====
    # find key points using the FAST detector
    kp1 = IF.Detect(imgG1, ravelOutputFlag=True, showKpFlag=True)
    
    # ===== Descriptor =====
    # get key points and the corresponding feature vectors which are produced by the ORB descriptor
    kp1, des1 = IF.Describe(imgG1, kp1, ravelInputFlag=False, ravelOutputFlag=True)
    
    # ===== Matcher =====
    # finds key points, their descriptions and applies the bfmatcher
    pts1, pts2 = IF.Match(imgG1, imgG2, showFlag=True)    
    """        
    
    def __init__(self, detectorMethod=['fast'], descriptorMethod=['orb'], matchingMethod='bfmatcher'):
        self._logger = _ImgFeatLogger
        self._keepLogFile = True # set this variable to False to disable the logger
        self._logger.Log(
            self._keepLogFile, 
            'info', 
            '\n\n############### NEW INSTANCE OF ImgFeatures CLASS STARTS HERE ###############'
        )

        self._logger.Log(self._keepLogFile, 'info', '===== CONSTRUCTOR =====')
        self._detectorMethod = detectorMethod
        self._descriptorMethod = descriptorMethod 
        self._matchingMethod = matchingMethod


        # ===== Detector ===============================================================================================
        # Supported detectors: orb, harris, brisk, fast, shitomasi, harrislaplace

        self._detector = []
        for detectorItem in detectorMethod:
            if detectorItem == 'orb': # fast enough
                self._detector.append(
                    _cv.ORB_create(nfeatures=5000, scaleFactor = 1.414, scoreType=_cv.ORB_FAST_SCORE)
                )
                self._logger.Log(self._keepLogFile, 'info', 'Selected Detector: ORB')
            elif detectorItem == 'harris': # very slow (the delay is owed to the keypoint production code)
                self._detector.append(_HarrisDetector)
                self._logger.Log(self._keepLogFile, 'info', 'Selected Detector: HARRIS')
            elif detectorItem == 'brisk': # slow
                self._detector.append(_cv.BRISK_create())
                self._logger.Log(self._keepLogFile, 'info', 'Selected Detector: BRISK')
            elif detectorItem == 'fast': # fastest indeed
                self._detector.append(_cv.FastFeatureDetector_create())
                self._logger.Log(self._keepLogFile, 'info', 'Selected Detector: FAST')
            elif detectorItem == 'shitomasi': # very slow
                self._detector.append(_ShiTomasiDetector)
                self._logger.Log(self._keepLogFile, 'info', 'Selected Detector: SHI-TOMASI')
            elif detectorItem == 'harrislaplace': # slowetst
                self._detector.append(
                    _cv.xfeatures2d.HarrisLaplaceFeatureDetector_create(2, maxCorners=5000)
                )
                self._logger.Log(self._keepLogFile, 'info', 'Selected Detector: HARRIS-LAPLACE')


        # ===== Descriptor =============================================================================================
        # Supported descriptors: orb, brisk

        self._descriptor = []
        for descriptorItem in descriptorMethod: 
            if descriptorItem == 'orb':
                self._descriptor.append(
                    _cv.ORB_create(nfeatures=5000, scaleFactor = 1.414, scoreType=_cv.ORB_FAST_SCORE)
                )
                self._logger.Log(self._keepLogFile, 'info', 'Selected Descriptor: ORB')
            elif descriptorItem == 'brisk':
                self._descriptor.append(_cv.BRISK_create())
                self._logger.Log(self._keepLogFile, 'info', 'Selected Descriptor: BRISK')


        # ===== Matcher ================================================================================================
        # Supported matchers: bfmathcer, flannmatcher

        if matchingMethod == 'bfmatcher':
            self._matcher = []
            for descriptorItem in descriptorMethod:
                if descriptorItem == 'orb' or descriptorItem == 'brisk' or descriptorItem == 'brief':
                    self._matcher.append(_cv.BFMatcher(_cv.NORM_HAMMING))
                else:
                    self._matcher.append(_cv.BFMatcher())
            self._logger.Log(self._keepLogFile, 'info', 'Selected Matcher: BFMatcher')

        elif matchingMethod == 'flannmatcher':
            self._matcher = []
            for descriptorItem in descriptorMethod:
                if descriptorItem == 'orb' or descriptorItem == 'brisk' or descriptorItem == 'brief':
                    FLANN_INDEX_LSH = 6
                    index_params = dict(
                        algorithm = FLANN_INDEX_LSH,
                        table_number = 6,
                        key_size = 12,
                        multi_probe_level = 1
                    )                
                else:
                    FLANN_INDEX_KDTREE = 1 # the online tutorial assigns the value 0, but it is not correct.
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
                search_params = dict(checks=50)   # or pass empty dictionary

                self._matcher.append(_cv.FlannBasedMatcher(index_params, search_params))
            self._logger.Log(self._keepLogFile, 'info', 'Selected Matcher: FLANNMatcher')
        self._logger.Log(self._keepLogFile, 'info', '---------------------------------------------------------------------------------------')

    
    def __del__(self):
        """Clears the logger.
        """
        self._logger.KillLogger()


    def Detect(self, img, **kwargs): #ravelOutputFlag=True, showKpFlag=False): 
        """Detects key points in the input image. 
        
        Parameters
        ---------- 
        img: ndarray
            The image in which we want to detect the key points. 
            
            NOTE: if the image has been read by using the opencv library, the collor channels are BGR
            instead of RGB! If needed, use the colorChannelsOrder parameter to inform the function about the 
            correct channels order.
            Example: 
                IF = ImgFeatures(detectorMethod=['fast', 'orb'])
                kp = IF.Detect(img, colorChannelsOrder='RGB')
        
        **kwargs: keyword argumets
            The Detect function, in its current form, can take three different keyword arguments.
            These arguments are:
                ravelOutputFlag                 
                showKpFlag
                colorChannelsOrder
                
            ravelOutputFlag: bool, (optional, default: True)   
                Taking into account that multiple detectors may be applied in the same image,
                the detected key points are organized in a list, in which each element keeps
                points for a particular detector. In case where this parameters is True, the 
                resulted output is converted to a single list where each element is a key point.
                If the parameter is False, the resulted output is a list where each element 
                is a list of key points.
                
            showKpFlag: bool, (optional, default: False)
                Set this flag to True to see the detection result. The key points can be seen in green dots.
                
            colorChannelsOrder: str, (optional, default: 'BGR')
                This parameter informs the Detect function regarding the images channel order. 
                Since the cvToolbox is currently based mostly on openCV, it is assumed that
                the input image follows the BGR channel order. 
            
        Returns 
        ------- 
        list of keyPoints or list of lists of keyPoints
            The function's output is affected by the values of keyword arguments. In case that
            ravelOutputFlag is True, the function results in a list of keyPoints. In case that
            ravelOutputFlag is False, the function results in a list of lists of keyPoints.
            
        Examples 
        -------- 
        IF = ImgFeatures(detectorMethod=['fast'], descriptorMethod=['orb'], matchingMethod='bfmatcher')
        img = _cv.imread('../data/calibData/left02.jpg')
        kp = IF.Detect(img, ravelOutputFlag=True, showKpFlag=True)
        """
        
        self._logger.Log(self._keepLogFile, 'info', '===== DETECT =====')
        _myTimer.tic('Detector')
        
        # ===== Check input image type =====
        if type(img) != _np.ndarray:
            self._logger.Log(self._keepLogFile, 'error', 'The input image must be of type numpy.ndarray.')
            return None
        
        
        # ===== Parse kwargs and define the default values =====
        KeyWords = {
            'ravelOutputFlag': True, 
            'showKpFlag': False, 
            'colorChannelsOrder': 'BGR',
        }
        
        for kw in KeyWords:
            if kw in kwargs:
                KeyWords[kw] = kwargs[kw]
        
        
        # ===== Check input image color type =====
        if len(img.shape) > 2:
            self._logger.Log(self._keepLogFile, 'info', 'The input image is NOT grayscale.')
            if KeyWords['colorChannelsOrder'] == 'BGR':
                self._logger.Log(
                    self._keepLogFile, 
                    'info', 
                    'The input image converted to grayscale considering the color channels order as BGR.'
                )
                img = _cv.cvtColor(img, _cv.COLOR_BGR2GRAY)        
            elif KeyWords['colorChannelsOrder'] == 'RGB':
                self._logger.Log(
                    self._keepLogFile, 
                    'info', 
                    'The input image converted to grayscale considering the color channels order as RGB.'
                )
                img = _cv.cvtColor(img, _cv.COLOR_RGB2GRAY)
            else:
                self._logger.Log(
                    self._keepLogFile, 
                    'info', 
                    "\n\tThe input image converted to grayscale considering the color\n" + 
                    "\tchannels order as BGR. If you want to define an RGB order use\n" + 
                    "\tthe keyward argument: colorChannelsOrder='RGB'"
                )
                img = _cv.cvtColor(img, _cv.COLOR_BGR2GRAY)   
        else:
            self._logger.Log(self._keepLogFile, 'info', 'The input image is grayscale.')
        
        
        # ===== Use the selected detectors to detect key points =====
        kp = []
        for detector in self._detector:
            kp.append(detector.detect(img, None))

            
        # ===== Ravel the key point list =====
        if KeyWords['ravelOutputFlag']:
            # TODO: must remove potential double keypoints coming out from different detectors
            kp = [kpnt for i in range(len(kp)) for kpnt in kp[i]]            
            self._logger.Log(self._keepLogFile, 'info', 'Key points list has been raveled')

        
        toc = _myTimer.toc('Detector')
        # ===== Show the detected points =====
        if KeyWords['showKpFlag']:
            if not KeyWords['ravelOutputFlag']:
                # TODO: must remove potential double keypoints coming out from different detectors
                kp_ = [kpnt for i in range(len(kp)) for kpnt in kp[i]]    
            else:
                kp_ = kp

            img_kp = _cv.drawKeypoints(img, kp_, None, color=(0,255,0), flags=_cv.DrawMatchesFlags_DEFAULT)
            _plt.figure()
            _plt.imshow(img_kp, cmap='gray')
            _plt.show()
            
        
        self._logger.Log(self._keepLogFile, 'info', 'Detector has finished in time: {0} msec'.format(toc))
        self._logger.Log(self._keepLogFile, 'info', '---------------------------------------------------------------------------------------')

        return kp    