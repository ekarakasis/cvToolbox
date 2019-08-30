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

import sys
sys.path.append('../')
sys.path.append('../../')

from cvToolbox.caLib import CameraCalibration

CalibImagePath = '../data/calibData/'
CalibImageType = '.jpg'
Image4Undistort = '../data/calibData/left02.jpg'

testID = 0   

def CheckResult(flag, testID):
    if flag:
        print('>>> example01 test: ' + str(testID) + ': OK')
    else: 
        print('>>> example01 test: ' + str(testID) + ': ERROR') 

def example01():
    cc = CameraCalibration()
    ret = cc.LoadImageNames(CalibImagePath, typeFilter=[CalibImageType])
    CheckResult(ret, testID)
    if ret:
        print(cc.GetImageNames())

if __name__ == '__main__':
    example01()