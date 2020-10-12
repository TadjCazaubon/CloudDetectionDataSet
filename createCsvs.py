__author__ = 'Tadj Cazaubon'
__credits__ = ["Tadj Cazaubon",'Tohid Ardeshiri']


import os
import cv2
import csv
import matplotlib
import numpy as np
import multiprocessing
import threading
import concurrent.futures
from datetime import datetime
from matplotlib import pyplot as plt




csv.field_size_limit(2147483647)

"""
First we need a file queing function to match our image files by name,
then we need to pass them to our pixel-sort function to generate our csvs.

We use Multiprocessing to simultaneously work with our files, then
Multithreading within each to write to our csv files.
"""




def create_csv(Blocked,Reference,counter,FuckedImagesCounter):


    """
    An assumption of this function is that all images are of the same size.
    As to not raise an error, I will resize them to 400px300p
    """



    """
    We define our globals we may need for later such as paths
    """


    i = counter


    cloudCsvFolder = r'Main\Cloud-CSVS'
    cloudCsvName = 'BGRDistribution.csv'
    cloudCsvPath = os.path.join(cloudCsvFolder,cloudCsvName)

    skyCsvFolder = r'Main\Sky-CSVS'
    skyCsvName = 'BGRDistribution.csv'
    skyCsvPath = os.path.join(skyCsvFolder,skyCsvName)

    ImageErrorLogs = r"Main/ErrorLogs/VisualisedImageErrors"


    cloudPixelsBGR = []
    skyPixelsBGR = []


    #----------------------------------------------------------------------------------------------------#


    """
    We read in our images and convert them to HSV
    """


    blockedImage = cv2.imread(Blocked)
    blockedImage = cv2.cvtColor(blockedImage,cv2.COLOR_BGR2HSV)
    #blockedImage = cv2.resize(blockedImage,(400,300))

    referenceImage = cv2.imread(Reference)
    referenceImage = cv2.cvtColor(referenceImage,cv2.COLOR_BGR2HSV)
    #referenceImage = cv2.resize(referenceImage,(400,300))


    #----------------------------------------------------------------------------------------------------#


    """
    First we make a mask for red colours
    We'll use Red to represent Clouds
    Red can have hue values between 0-10, but also 170-180
    """


    u_b_red1 = np.array([10, 255, 255])
    l_b_red1 = np.array([0, 30, 30])

    u_b_red2 = np.array([180, 255, 255])
    l_b_red2 = np.array([170, 50, 50])

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


    #----------------------------------------------------------------------------------------------------#


    """
    Now we use apply masks via bitwise ands to get our cloud and sky
    """


    cloudImage = cv2.bitwise_and(referenceImage,referenceImage,mask = redMask)
    skyImage = cv2.bitwise_and(referenceImage,referenceImage,mask = blackMask)



    #cv2.imshow('BlackMask',blackMask)
    #cv2.imshow('RedMask',redMask)
    #cv2.imshow('referenceImage',referenceImage)
    #cv2.imshow('blockedImage',blockedImage)

    #cv2.waitKey(0)

    #----------------------------------------------------------------------------------------------------#

    """
    Saving the Images to see where I fucked up
    """






    """blockedImageName = f"Blocked{FuckedImagesCounter}.jpg"
    blockedImageErrorPath = os.path.join(ImageErrorLogs,blockedImageName)
    blockedImageError = cv2.cvtColor(blockedImage,cv2.COLOR_HSV2BGR)
    cv2.imwrite(blockedImageErrorPath,blockedImageError)"""

    """referenceImageName = f"Reference{FuckedImagesCounter}.jpg"
    referenceImageErrorPath = os.path.join(ImageErrorLogs,referenceImageName)
    referenceImageError = cv2.cvtColor(referenceImage,cv2.COLOR_HSV2BGR)
    cv2.imwrite(referenceImageErrorPath,referenceImageError)"""

    cloudImageName = f"Cloud{FuckedImagesCounter}.jpg"
    cloudeImageErrorPath = os.path.join(ImageErrorLogs,cloudImageName)
    cloudImageError = cv2.cvtColor(cloudImage,cv2.COLOR_HSV2BGR)
    cv2.imwrite(cloudeImageErrorPath,cloudImageError)

    skyImageName = f"Sky{FuckedImagesCounter}.jpg"
    skyImageErrorPath = os.path.join(ImageErrorLogs,skyImageName)
    skyImageError = cv2.cvtColor(skyImage,cv2.COLOR_HSV2BGR)
    cv2.imwrite(skyImageErrorPath,skyImageError)

    #----------------------------------------------------------------------------------------------------#


    """
    We iterate through our two masked images using zip() and take note of the values
    of pixels that aren't true black (0,0,0) by saving them in lists
    """

    #cloudImage = cv2.cvtColor(cloudImage,cv2.COLOR_HSV2BGR)
    #skyImage = cv2.cvtColor(skyImage,cv2.COLOR_HSV2BGR)

    for cloudPixel,skyPixel in zip(cloudImage,skyImage):
        for cloudPixelValue,skyPixelValue in zip(cloudPixel,skyPixel):

            cloudB,cloudG,cloudR = cloudPixelValue
            skyB,skyG,skyR = skyPixelValue

            if cloudB!=0 or cloudG!=0 or cloudR!=0:
                cloudPixelsBGR.append(list(cloudPixelValue))
            if skyB!=0 or skyG!=0 or skyR!=0:
                skyPixelsBGR.append(list(skyPixelValue))

            else:
                continue


    #----------------------------------------------------------------------------------------------------#


    """
    We have to create our target function to dynamically write to our csv and report if an error
    occurs.File data that cannot be written to should be reported so they can be ommitted.

    Future versions should have an error log to write to after each attempted run.
    """

    def writeCsv(filePath,pixelBGR):
        try:
            with open(filePath,'a+',newline = '') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerows(pixelBGR)
                print("Writing to csv")
            csvFile.close()

        except Exception as exception:

            DataValueErrorLogName = (datetime.now().strftime("%Y_%b_%d_%H-%M-%S") +".txt")
            DataValueErrorLogPath = (r"Main/ErrorLogs/WriteLogs/"+ DataValueErrorLogName)
            logfile = open(DataValueErrorLogPath,"a")
            errorlog = f"Corrupted or missing value in the form {pixelBGR}"
            logfile.write(errorlog)

            logfile.close()



    """
    Now we create two threads to write to our csv files concurrrently
    """



    skyCsvWriter = threading.Thread(target = writeCsv, args = [skyCsvPath,skyPixelsBGR])
    cloudCsvWriter = threading.Thread(target = writeCsv, args = [cloudCsvPath,cloudPixelsBGR])

    skyCsvWriter.start()
    cloudCsvWriter.start()

    skyCsvWriter.join()
    cloudCsvWriter.join()


#---------------------------------------------------------------------------------------------------------#

def distributionBarGraphGenerator(cloudCsvFolder,skyCsvFolder,graphFolder,distributionCsvName,bins):


    """
    We create our Green, Blue, and Red Distributions here.
    """

    cloudDistributionCsvPath = os.path.join(cloudCsvFolder,distributionCsvName)
    skyDistributionCsvPath = os.path.join(skyCsvFolder,distributionCsvName)
    bgrGraphsavePath = os.path.join(graphFolder,'HSVBarGraph.pdf')


    def readDistributionData(distributionCsv):
        blues = []
        greens = []
        reds = []

        with open(distributionCsv,'r') as file:

            DataValueErrorLogName = (datetime.now().strftime("%Y_%b_%d_%H-%M-%S")+".txt")
            DataValueErrorLogNamePath = r"Main/ErrorLogs/ReadLogs/"+ DataValueErrorLogName
            logfile = open(DataValueErrorLogNamePath,'a+')

            csvReader = csv.reader(file)

            for row in csvReader:
                try:
                    blues.append(int(row[0]))
                    greens.append(int(row[1]))
                    reds.append(int(row[2]))
                    #BGRListData.append([int(lineList[0]),int(lineList[1]),int(lineList[2])])

                except Exception as e:
                    errorlog = f'\nCorrupted data in the form {row} in {distributionCsv}'
                    logfile.write(errorlog)
                    #print(errorlog)

        file.close()

        return (blues,greens,reds)




    """
    We get the bgr values from our csvs simultaneously and then pass this
    to the graph data
    """


    with concurrent.futures.ThreadPoolExecutor() as executor:

        cloudDistributionDataThread = executor.submit(readDistributionData,cloudDistributionCsvPath)
        skyDistributionDataThread = executor.submit(readDistributionData,skyDistributionCsvPath)

        cloudDistributionBGR = cloudDistributionDataThread.result()
        skyDistributionBGR = skyDistributionDataThread.result()


        cloudBlues,cloudGreens,cloudReds = cloudDistributionBGR
        skyBlues,skyGreens,skyReds = skyDistributionBGR



    fig,axes = plt.subplots(nrows = 3,ncols = 1)
    axes = axes.flatten()

    axes[0].hist(cloudBlues, bins = bins,color = 'blue',alpha= 0.3,label = 'Cloud Hues')
    axes[0].hist(skyBlues,bins = bins,color = 'purple',alpha = 0.3,label = 'Sky Hues')
    axes[0].set_xlabel('HSV Hue (0 - 255)')
    axes[0].set_ylabel('frequency')
    axes[0].legend(loc="upper left")

    axes[1].hist(cloudGreens, bins = bins,color = 'green',alpha= 0.3,label = 'Cloud Saturation')
    axes[1].hist(skyGreens,bins = bins,color = 'yellow',alpha = 0.3,label = 'Sky Saturation')
    axes[1].set_xlabel('HSV Saturation Value (0 - 255)')
    axes[1].set_ylabel('frequency')
    axes[1].legend(loc="upper left")

    axes[2].hist(cloudReds, bins = bins,color = 'red',alpha= 0.3,label = 'Cloud Light Values')
    axes[2].hist(skyReds,bins = bins,color = 'pink',alpha = 0.3,label = 'Sky Light Values')
    axes[2].set_xlabel('HSV Light Value (0 - 255)')
    axes[2].set_ylabel('frequency')
    axes[2].legend(loc="upper left")

    fig.tight_layout()
    plt.savefig(bgrGraphsavePath)


#---------------------------------------------------------------------------------------------------------#


"""
This is our main function area.
"""

def main():


    """
    Space for important variables
    """
    bins = [*range(0,256,1)]
    FuckedImagesCounter = 0
    blockedImageFolder = r'Main\Blocked-Images'
    #blockedImageName = r''
    #blockedImagePath = os.path.join(blockedImageFolder,blockedImageName)

    referenceImageFolder = r'Main\Reference-Images'
    #referenceImageName = r''
    #referenceImagePath = os.path.join(referenceImageFolder,referenceImageName)


    activeThreads = []
    activeProcesses = []

    imagePairList=[]

    counter = 0


    cloudCsvFolder = r'Main\Cloud-CSVS'
    skyCsvFolder = r'Main\Sky-CSVS'
    graphFolder = r'Main\Graphs'

    distributionCsvName = 'BGRDistribution.csv'

#---------------------------------------------------------------------------------------------------------#
    """
    If our distribution csvs already exist, we need to delete them
    """

    cloudDistributionCsvPath = os.path.join(cloudCsvFolder,distributionCsvName)
    skyDistributionCsvPath = os.path.join(skyCsvFolder,distributionCsvName)

    if os.path.isfile(cloudDistributionCsvPath):
        os.remove(cloudDistributionCsvPath)
    if os.path.isfile(skyDistributionCsvPath):
        os.remove(skyDistributionCsvPath)

#---------------------------------------------------------------------------------------------------------#


    """
    Firstly we need to match our blocked images to our reference images of the
    same name in two seperate folders

    We perform os.walk() on our reference image, then walkthrough our blocked
    images for each reference and match the file names. If the names match, they
    are added as a key-value pair in a dictionary and added to our list of pairs.

    We can make this faster by using our multiprocessing library to make each os.walk() a thread.
    We use threads over processes here because this task is io bound
    """


    def fileScan(arguments):
        imagePairSinglet = []
        blockedImageName,blockedImageFolder,referenceImageFolder=arguments

        """
        We perform an os.walk() on our reference image and compare to our passed in
        blocked image
        """


        for referenceImagesRoot,referenceImagesFolders,referenceImages in os.walk(referenceImageFolder):
            for referenceImageName in referenceImages:
                if str(blockedImageName) == str(referenceImageName):
                    blockedImagePath = os.path.join(blockedImageFolder,blockedImageName)
                    referenceImagePath = os.path.join(referenceImageFolder,referenceImageName)
                    imagePairSinglet.append([blockedImagePath,referenceImagePath])

                else:
                    continue
        return imagePairSinglet



    with concurrent.futures.ThreadPoolExecutor() as executor:
        for blockedImagesRoot,blockedImagesFolders,blockedImages in os.walk(blockedImageFolder):
            for blockImageName in blockedImages:
                future = executor.submit(fileScan, [blockImageName,blockedImageFolder,referenceImageFolder])
                returnValue = (future.result())
                imagePairList.append(returnValue)
                print(f'found image pair: {returnValue}')


#---------------------------------------------------------------------------------------------------------#

    """
    Now we iterate through our imagePairList and call create_csv for each
    """

    for imagePair in imagePairList:
        try:
            print(f'\nNow appending images: \n > {imagePair[0]}')
            blockedImagePath,referenceImagePath = imagePair[0]
            counter+=1
            FuckedImagesCounter+=1
            arguments = [blockedImagePath,referenceImagePath,counter,FuckedImagesCounter]

            p = multiprocessing.Process(target = create_csv,args = arguments)
            p.start()
            activeProcesses.append(p)

        except Exception:
            continue


    for process in activeProcesses:
        process.join()


#--------------------------------------------------------------------------------------------------------------#

    """
    Now we create our BGR Bar Graph
    """
    distributionBarGraphGenerator(cloudCsvFolder,skyCsvFolder,graphFolder,distributionCsvName,bins)


if __name__ == '__main__':
    main()
