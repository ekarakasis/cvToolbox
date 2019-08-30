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

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#                   Unit testing
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

import sys
sys.path.append('../')
sys.path.append('../../')

import unittest
import cv2
from cvToolbox.imgFeat import ImgFeatures


def ApplyDetector(img, **kwargs):
    IF = ImgFeatures(detectorMethod=kwargs['detectorMethod'])    
    
    if kwargs['imgType'] == 'gray':
        imgP = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        imgP = img
        
    kp = IF.Detect(
        imgP, 
        ravelOutputFlag=kwargs['ravelOutputFlag'], 
        showKpFlag=kwargs['showKpFlag'], 
        colorChannelsOrder=kwargs['colorChannelsOrder']
    )
    return kp

img1 = cv2.imread('G:/001 Work/portfolio/#objectPointCloud/calibdata/fromopencv/left02.jpg')
detectorParams={
    'imgType': 'gray',
    'detectorMethod': ['fast'], 
    'ravelOutputFlag': True, 
    'showKpFlag': False, 
    'colorChannelsOrder': 'BGR',
}

class TestImgFeature(unittest.TestCase):        
    def test_1_detect_image(self): 
        
        for imgType in ['gray', 'rgb']:            
            detectorParams['imgType'] = imgType
            kp = ApplyDetector(img1, **detectorParams)

            self.assertEqual(type(kp), list)
            self.assertEqual(type(kp[0]), cv2.KeyPoint)
            self.assertEqual(type(kp[0].pt[0]), float)
        
    def test_2_detect_detectMethod(self):        
        
        for imgType in ['gray', 'rgb']: 
            for detectorMethod in ['fast', 'harris', 'orb', 'brisk', 'shitomasi', 'harrislaplace']:
                detectorParams['imgType'] = imgType
                detectorParams['detectorMethod'] = [detectorMethod]
                
                # print(detectorParams)
                kp = ApplyDetector(img1, **detectorParams)

                self.assertEqual(type(kp), list)
                self.assertEqual(type(kp[0]), cv2.KeyPoint)
                self.assertEqual(type(kp[0].pt[0]), float)
                
    def test_3_detect_ravelOutputFlag(self):        
        detectorParams['ravelOutputFlag'] = True
        for imgType in ['gray', 'rgb']: 
            for detectorMethod in ['fast', 'harris', 'orb', 'brisk', 'shitomasi', 'harrislaplace']:
                detectorParams['imgType'] = imgType
                detectorParams['detectorMethod'] = [detectorMethod]
                
                kp = ApplyDetector(img1, **detectorParams)

                self.assertEqual(type(kp), list)
                self.assertEqual(type(kp[0]), cv2.KeyPoint)
                self.assertEqual(type(kp[0].pt[0]), float)
                
        detectorParams['ravelOutputFlag'] = False
        for imgType in ['gray', 'rgb']: 
            for detectorMethod in ['fast', 'harris', 'orb', 'brisk', 'shitomasi', 'harrislaplace']:
                detectorParams['imgType'] = imgType
                detectorParams['detectorMethod'] = [detectorMethod]
                
                kp = ApplyDetector(img1, **detectorParams)

                self.assertEqual(type(kp), list)
                self.assertEqual(type(kp[0]), list)
                self.assertEqual(type(kp[0][0]), cv2.KeyPoint)
                self.assertEqual(type(kp[0][0].pt[0]), float)
                
    def test_4_detect_colorChannelsOrder(self):        
        detectorParams['ravelOutputFlag'] = True
        for imgType in ['gray', 'rgb']: 
            for detectorMethod in ['fast', 'harris', 'orb', 'brisk', 'shitomasi', 'harrislaplace']:
                for colorChannelsOrder in ['RGB', 'BGR']:
                    detectorParams['imgType'] = imgType
                    detectorParams['detectorMethod'] = [detectorMethod]
                    detectorParams['colorChannelsOrder'] = colorChannelsOrder

                    kp = ApplyDetector(img1, **detectorParams)

                    self.assertEqual(type(kp), list)
                    self.assertEqual(type(kp[0]), cv2.KeyPoint)
                    self.assertEqual(type(kp[0].pt[0]), float)
                
        detectorParams['ravelOutputFlag'] = False
        for imgType in ['gray', 'rgb']: 
            for detectorMethod in ['fast', 'harris', 'orb', 'brisk', 'shitomasi', 'harrislaplace']:
                for colorChannelsOrder in ['RGB', 'BGR']:
                    detectorParams['imgType'] = imgType
                    detectorParams['detectorMethod'] = [detectorMethod]
                    detectorParams['colorChannelsOrder'] = colorChannelsOrder
                
                    kp = ApplyDetector(img1, **detectorParams)

                    self.assertEqual(type(kp), list)
                    self.assertEqual(type(kp[0]), list)
                    self.assertEqual(type(kp[0][0]), cv2.KeyPoint)
                    self.assertEqual(type(kp[0][0].pt[0]), float)


if __name__ == '__main__':
    unittest.main()                    