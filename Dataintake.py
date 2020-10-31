__author__ = 'Tadj Cazaubon'
__credits__ = ['Tadj Cazaubon','Tohid Ardeshiri']


import os
import cv2
import matplotlib
import numpy as np
import multiprocessing
import threading
import concurrent.futures
from datetime import datetime
from matplotlib import pyplot as plt

"""
First we need a file queuing function to match our image files by name, then we
need to pass them to our pixel-sorting function to read in our cloud and sky
pixel values. We use Multiprocessing to simultaneously work with multiple files,
then multithreading for writing these values to our database.
"""

def readImagesIntoDataBase(blockedImagePath,referenceImagePath,counter):

    """
    Images are no longer resized as of this revamp. It is preferred to have
    more, lower quality images of varying conditions if possible.
    """



    """
    Varibales we need later within the function
    """






    #---------------------------------------------------------------------------

    "We read in our images and convert them to HSV, keeping a copy in BGR format."

    blockedImageBGR = cv2.imread(blockedImagePath)
    blockedImageHSV = cv2.cvtColor(blockedImageBGR,cv2.COLOR_BGR2HSV)
    #blockedImage = cv2.resize(blockedImage,(400,300))

    referenceImageBGR = cv2.imread(referenceImagePath)
    referenceImageHSV = cv2.cvtColor(referenceImageBGR,cv2.COLOR_BGR2HSV)
    #referenceImage = cv2.resize(referenceImage,(400,300))

    #---------------------------------------------------------------------------

        """
        First we make a mask for red colours
        We'll use Red to represent Clouds
        Red can have hue values between 0-10, but also 170-180
        """

        u_b_red1 = np.array([10, 255, 255])
        l_b_red1 = np.array([0, 30, 30])

        u_b_red2 = np.array([180, 255, 255])
        l_b_red2 = np.array([170, 50, 50])

        """
        Technically, if the red we used is all within one range, we should have
        no need to do this sort of thing, bu tthat is to be looked at later.
        """
        maskOneRed = cv2.inRange(blockedImage,l_b_red1,u_b_red1)
        maskTwoRed = cv2.inRange(blockedImage,l_b_red2,u_b_red2)

        redMask = cv2.bitwise_or(maskOneRed,maskTwoRed)



        """
        Now we do the same for Black.
        We'll use a range of black to represent The Sky
        """

        u_b_black = np.array([180, 255,30])
        l_b_black = np.array([0, 0, 0])

        blackMask = cv2.inRange(blockedImage,l_b_black,u_b_black)

        """
        This chosen range is chosen purely due to trial and error and
        a single stackoverflow question.
        """

        #----------------------------------------------------------------------------------------------------#


        """
        Now we apply masks via bitwise ands to get our cloud and sky
        """


        cloudImageHSV = cv2.bitwise_and(referenceImageHSV,referenceImageHSV,mask = redMask)
        cloudImageBGR = cv2.cvtColor(cloudImageHSV,cv2.COLOR_HSV2BGR)

        skyImageHSV = cv2.bitwise_and(referenceImageHSV,referenceImageHSV,mask = blackMask)
        skyImageBGR = cv2.cvtColor(skyImageHSV,cv2.COLOR_HSV2BGR)


        #---------------------------------------------------------------------------


        """
        In a previous version, we have used a zip to iterate through our masked
        images to note their pixel values. Here , since we have four images to
        iterate through, I will iterate through the clouds and sky image sets
        as two separate zips I am fully aware that I can still use a zip, but to avoid
        both confusion and possible error with the image iteration, I will do it
         this way.
        """

        for HSVpixel,BGRpixel  in zip(cloudImageHSv,cloudImageBGR):
            """
            We can now iterate through both simultaneously and append each to a
            respective array
            """
            for HSVpixelValue, BGRpixelValue in zip(HSVpixel,BGRpixel) 
