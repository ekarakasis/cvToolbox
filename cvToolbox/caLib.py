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

import os as _os
import cv2 as _cv
import datetime as _datetime
import pickle as _pickle
import numpy as _np 
import scipy.io as _sio
from Logger.Logger import Logger as _Logger

# ===== Utilities ===================================================================

def _MkDir(rootFolder):
    if not _os.path.exists(rootFolder):
        _os.mkdir(rootFolder)
        
# def DelFile(fileName):
#     if _os.path.exists(fileName):
#         _os.remove(fileName)

# ===== Calibration Logger ==========================================================

# ------------
# Kill all previous loggers and delete the corresponding log files  
# BE CAREFUL!!! this lines may kill and delete other useful loggers and 
# log files in your application.
# _Logger.KillAllLoggers()
# _Logger.DeleteAllLogFiles()
# ------------

class _CalibLogger: 
    TNLM = _Logger('calibLogger.TNLM', 'calibrationLog', 'Logs', 'time-name-level-message')
    Unformated = _Logger('calibLogger.U', 'calibrationLog', 'Logs', 'unformated')
    
    @staticmethod
    def Log(logFlag, logLevel, *args):
        _CalibLogger.TNLM.Log(logFlag, logLevel, *args)
                
    @staticmethod
    def Custom(logFlag, logLevel, *args):
        _CalibLogger.Unformated.Log(logFlag, logLevel, *args)
        
    @staticmethod
    def KillLogger():
        _CalibLogger.TNLM.CloseLogger()
        _CalibLogger.Unformated.CloseLogger()

    @staticmethod
    def KillLogger_n_DeleteLogFiles():
        _CalibLogger.TNLM.CloseLogger()
        _CalibLogger.Unformated.CloseLogger()
        
        _CalibLogger.TNLM.DeleteLogFile()
        _CalibLogger.Unformated.DeleteLogFile()


# ===== Camera Calibration Class ====================================================

class CameraCalibration: 
    """A simple class that can be use to calibrate a camera. 
    
    Parameters 
    ----------  
    calibParamsFileName: str (optional, default: None)
        The name of the output file in which we want to save the camera matrix
        and the distortion coefficients (i.e. calibration parameters).
    """
        
    def __init__(self, calibParamsFileName=None):                    
        self._logger = _CalibLogger
        self._keepLogFile = True 
        self._logger.Custom(
            self._keepLogFile, 
            'info', 
            '\n\n############### NEW INSTANCE OF CameraCalibration CLASS STARTS HERE ###############'
        )
            
        self._imageNames = None
        self._imagesPath = None
        self._imgShape = None
        self._calibParamsFileName = calibParamsFileName

        # termination criteria when finding corners with subpixel accuracy
        self._criteria = (_cv.TERM_CRITERIA_EPS + _cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self._objpoints = None # 3d point in real world space
        self._imgpoints = None # 2d points in image plane.

        self._mtx, self._dist = None, None                        
        

    def __del__(self):
        """Clears the logger.
        """
        self._logger.KillLogger()


    def GetImageNames(self):
        """Returns the list of image names. 
        """
        self._logger.Custom(self._keepLogFile, 'info', "\n===== GetImageNames =====")
                                        
        if self._imageNames is None:
            self._logger.Log(self._keepLogFile, 'warning', "The imageNames list has not initialized yet.")
        else:
            self._logger.Log(self._keepLogFile, 'info', "The imageNames list includes records.")
            
        return self._imageNames
                

    def LoadImageNames(self, imagesPath, typeFilter=('.jpg', '.png', '.tif')):
        """Searches in a specific folder in order to locate image files.
        
        Parameters 
        ---------- 
        imagesPath: str
            The root folder in which the images have been saved.
            
        typeFilter: tuple (optional, default: ('.jpg', '.png'))
            A tuple which keeps the types of supported images. 
            
        Returns 
        ------- 
        bool: 
            The funtion returns a boolean variable as an indicator of successfull run.
        """
         
        self._logger.Custom(self._keepLogFile, 'info', "\n===== LoadImageNames =====")
            
        self._imagesPath = imagesPath
        self._imageNames = []
        dirLst = _os.listdir(imagesPath)

        for item in dirLst:
            for flt in typeFilter:
                if flt in item:
                    self._imageNames.append(item)
                    self._logger.Log(self._keepLogFile, 'info', 'Loaded image with name {}.'.format(item))
                      
        if self._imageNames != []:
            self._logger.Log(self._keepLogFile, 'info', 'Image names successfully loaded')
            return True
        else:
            self._logger.Log(self._keepLogFile, 'error', 'Image names have not been loaded successfully.')
            return False              


    def FindChessboard(self, chessboardSize=(7, 9), squareSize=20, showImagesFlag=False):
        """Finds the chessboard in the images used for calibration. 
        
        Parameters  
        ---------- 
        chessboardSize: tuple, (optional, default: (7, 9))
            The size of the chessboard used in the calibration procedure. 
            
        squareSize: float, (optional, default: 20)
            The size in mm of each square in the chessboard.
            
        showImagesFlag: bool, (optional, default: False)
            A flag which determines whether or not you want to see the detected chessboards.
            
        Returns 
        ------- 
        bool: 
            The funtion returns a boolean variable as an indicator of successfull run.
        """
        
        self._logger.Custom(self._keepLogFile, 'info', "\n===== FindChessboard =====")
        
        if self._imageNames is not None and self._imagesPath is not None:        
            self._objpoints = [] # 3d point in real world space
            self._imgpoints = [] # 2d points in image plane.

            # prepare object points, like (0,0,0), (20,0,0), (40,0,0) ....,(120,100,0)
            objp = _np.zeros((chessboardSize[0]*chessboardSize[1],3), _np.float32)
            objp[:,:2] = _np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2) * squareSize

            for imgName in self._imageNames:

                fullPath = self._imagesPath + imgName
                img_c = _cv.imread(fullPath)
                img_g = _cv.cvtColor(img_c, _cv.COLOR_BGR2GRAY)                                

                ret, corners = _cv.findChessboardCorners(img_g, chessboardSize, None)

                # If found, add object points, image points (after refining them)
                if ret == True:                                        
                    self._objpoints.append(objp)

                    corners2 = _cv.cornerSubPix(img_g, corners, (11,11), (-1,-1), self._criteria)
                    self._imgpoints.append(corners2)
                    
                    self._logger.Log(self._keepLogFile, 'info', "The chessboard has been successfully detected in image %s.", imgName)

                    # Draw and display the corners
                    if showImagesFlag:
                        img = _cv.drawChessboardCorners(img_c, chessboardSize, corners2,ret)
                        _cv.imshow('img',img)
                        _cv.waitKey(350) 
                else:
                    self._logger.Log(self._keepLogFile, 'warning', "The chessboard has NOT been successfully detected in image %s.", imgName)

            self._imgShape = img_g.shape
            
            if showImagesFlag:
                _cv.destroyAllWindows()
                
            self._logger.Log(self._keepLogFile, 'info', "The chessboards finding process has been completed.")
            return True
        else:
            self._logger.Log(self._keepLogFile, 'warning', "No images have been detected yet. Please use first the function '.LoadImageNames()'.")
            return False
        
        return False


    def Calibration(self):
        """Starts the calibration process. 
        
        First we must have found the chessboard corners in all the used images.
        To get the parameters use the function: '.GetCalibrationParams()'
        
        Returns 
        ------- 
        bool: 
            The funtion returns a boolean variable as an indicator of successfull run.
        """
        
        self._logger.Custom(self._keepLogFile, 'info', "\n===== Calibration =====")
        
        if self._imgShape is not None and self._imgpoints is not None and self._objpoints is not None:
            # mtx: intrinsic camera matrix
            # dist: distortion coefficients
            # rvecs: rotation matrix
            # tvecs: translation matrix
            ret, self._mtx, self._dist, rvecs, tvecs = _cv.calibrateCamera(
                self._objpoints, 
                self._imgpoints, 
                self._imgShape[::-1], 
                None, 
                None
            )
            self._logger.Log(self._keepLogFile, 'info', "The calibration process has finished successfully.")
            return True
        else:
            self._logger.Log(self._keepLogFile, 'warning', "The chessboad corners have not been found yet.")
            return False
        
        return False


    def Undistort(self, img):
        """Undistorts the input image.
        
        The calibration parameters must have been calculated first.

        consider using the functions: 
            * getOptimalNewCameraMatrix or
            * initUndistortRectifyMap & remap
        to eliminate unwanted (black) pixels
        
        Parameters 
        ---------- 
        img: ndarray
            The image for which we want to remove the lest distortion.
        
        Returns  
        ------- 
        ndarray 
            Returns the processed (no black areas exist) undistorted image.
        """
        
        self._logger.Custom(self._keepLogFile, 'info', "\n===== Undistort =====")
        
        if self._mtx is not None:
            # consider using the functions: 
            #     * getOptimalNewCameraMatrix or
            #     * initUndistortRectifyMap & remap
            # to eliminate unwanted (black) pixels
            udst = _cv.undistort(img, self._mtx, self._dist, None, self._mtx)

            return udst
        elif self._mtx is None:
            self._logger.Log(self._keepLogFile, 'warning', "The camera matrix has not been calculated yet.")
        
        self._logger.Log(self._keepLogFile, 'error', "The lens distortion have not been removed.")
        return None


    def GetCalibrationParams(self):
        """Returns the a tuple with the camera matrix and the distortion coefficients.
        
        if the calibration parameters are not calculated, then the tuple includes None values.
        """
        
        self._logger.Custom(self._keepLogFile, 'info', "\n===== GetCalibrationParams =====")
        
        if self._mtx is not None:
            return self._mtx, self._dist
        return None, None


    def SaveParameters(self, fileName=None, save2MatFileFlag=False):
        """Saves the camera matrix and the distortion coefficients. 
        
        Parameters
        ---------- 
        fileName: str (optional)
            The desired name of the file. The default value is None, which means that the file name
            will be determined by the class constructor. In case that no name will be given in the
            constructor and in the function, then calibration parameters are saved in a file with name 
            that is given by the following pattern: "%Y-%m-%d_%H.%M.%S"_calibration'.
            
            NOTE: the 'fileName' has higher priority than the name that has been determined
            in the class constructor!
            
        save2MatFileFlag: bool (optional)
            A flag that determines the type of the file. The default value is False, which means
            that the calibration parameters will be saved in a 'pickle' file. If the flag is True
            the camera matrix and the distortion coefficients will be saved in a 'mat' file. 
            
        Returns
        -------
        bool: 
            The funtion returns a boolean variable as an indicator of successfull %run.
        """
        
        self._logger.Custom(self._keepLogFile, 'info', "\n===== SaveParameters =====")
        
        rootFolder = 'calibParams'
        _MkDir(rootFolder)
        
        # NOTE: the filename has higher priority than the name that has been determined
        # in the class initialization!
        
        if self._mtx is not None:
            if fileName is not None:                
                if save2MatFileFlag:
                    _sio.savemat(rootFolder + '/' + fileName + '.mat', {'calibrationParameters': [self._mtx, self._dist]})
                    _type = 'mat'
                else:
                    _type = 'pkl'
                    with open(rootFolder + '/' + fileName + '.pkl', 'wb') as f:
                        _pickle.dump([self._mtx, self._dist], f)
                
                self._logger.Log(self._keepLogFile, 'info', "Calibration parameters have been successfully saved with name: {0}.{1}.".format(fileName, _type))
                return True
            
            elif self._calibParamsFileName is not None:                
                if save2MatFileFlag:
                    _sio.savemat(rootFolder + '/' + self._calibParamsFileName + '.mat', {'calibrationParameters': [self._mtx, self._dist]})
                    _type = 'mat'
                else:
                    _type = 'pkl'
                    with open(rootFolder + '/' + self._calibParamsFileName + '.pkl', 'wb') as f:
                        _pickle.dump([self._mtx, self._dist], f)
                        
                self._logger.Log(self._keepLogFile, 'info', "Calibration parameters have been successfully saved with name: {0}.{1}.".format(self._calibParamsFileName, _type))
                return True
            
            else:                
                fnm = _datetime.datetime.today().strftime("%Y-%m-%d_%H.%M.%S") + '_calibration'
                if save2MatFileFlag:
                    _sio.savemat(rootFolder + '/' + fnm + '.mat', {'calibrationParameters': [self._mtx, self._dist]})
                    _type = 'mat'
                else:
                    _type = 'pkl'
                    with open(rootFolder + '/' + fnm + '.pkl', 'wb') as f:
                        _pickle.dump([self._mtx, self._dist], f)
                                    
                self._logger.Log(self._keepLogFile, 'info', "Calibration parameters have been saved successfully with name: {0}.{1}.".format(fnm, _type))
                return True
        else:
            self._logger.Log(self._keepLogFile, 'warning', "No calibration parameters have been calculated yet. Please consider run the '.Calibration()' function.")

        return False


    def LoadParameters(self, fileName=None):
        """Loads the camera matrix and the distortion coefficients.
        
        The function supports both 'pickle' and 'mat' files. In case of successfully loaded parameters,
        the result is not None. 
        
        Parameters 
        ---------- 
        fileName: str (optional)
            The desired name of the file. The default value is None, which means that the file name
            will be determined by the class constructor. In case that no name has been given in the
            constructor and the fileName is None, the function cannot load any file. 
            
            NOTE: the 'fileName' has higher priority than the name that has been determined
            in the class constructor!
            
            NOTE: In case that fileName is None then the 'pickle' file has higher priority than the 'mat' one 
            and thus, the function searches first for a pickle file and if it can't find anything then it 
            searches for a mat one!
            
        Returns
        ------- 
        tuple: (mtx, dist)
            The function results in the a tuple with the camera matrix and the distortion coefficients. 
            In case where the function cannot successfully load any file the tuple's values are both None.
        """
        
        self._logger.Custom(self._keepLogFile, 'info', "\n===== LoadParameters =====")
        
        rootFolder = 'calibParams'        
        
        mtx, dist = None, None

        # NOTE: the filename has higher priority than the name that has been determined
        # in the class initialization!
        
        if fileName is not None: 
            fname = fileName
            if fileName[-3:] == 'pkl':                
                with open(rootFolder + '/' + fileName, 'rb') as f:
                    mtx, dist = _pickle.load(f)
            elif fileName[-3:] == 'mat':
                mtx, dist = _sio.loadmat(rootFolder + '/' + fileName)['calibrationParameters'][0]
            else:
                self._logger.Log(self._keepLogFile, 'warning', "The file's type is not known. The file must have one of the following types: <fileName>.pkl or <fileName>.mat.")

        elif self._calibParamsFileName is not None:
            try:
                fname = self._calibParamsFileName + '.pkl'
                with open(rootFolder + '/' + self._calibParamsFileName + '.pkl', 'rb') as f:
                    mtx, dist = _pickle.load(f)
            except:
                try:
                    fname = self._calibParamsFileName + '.mat'
                    mtx, dist = _sio.loadmat(rootFolder + '/' + self._calibParamsFileName + '.mat')['calibrationParameters'][0]
                except:
                    self._logger.Log(self._keepLogFile, 'warning', "The file's type is not known. The file must have one of the following types: <fileName>.pkl or <fileName>.mat.")
        else:
            self._logger.Log(self._keepLogFile, 'warning', "No name has been determined. The function cannot load the calibration parameters.")

        if mtx is not None:
            self._logger.Log(self._keepLogFile, 'info', "Existed calibration parameters have been loaded successfully from the file {}.".format(fname))

        return mtx, dist 


    def SetCameraMatrix(self, cameraMat):
        """Sets the inputted camera matrix to the corresponding class variable.
        
        Parameters
        ----------
        cameraMat: ndarray
            The 3x3 camera matrix with which we want to update the current one.
            The shape of the ndarray must be (3,3).
        """
        self._mtx = _np.array(cameraMat)

    
    def SetDistortionCoefficients(self, distCoeffs):
        """Sets the inputted distortion coefficients to the corresponding class variable.
        
        Parameters
        ----------
        distCoeffs: ndarray
            The 1x5 vector with which we want to update the current one.
            The shape of the ndarray must be (1,5).
        """
        self._dist = _np.array(distCoeffs)