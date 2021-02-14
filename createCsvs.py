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

    cloudCSVFolder = r'Cloud-CSVS'
    cloudBGRCSVName = 'BGRDistribution.csv'
    cloudHSVCSVName="HSVDistribution.csv"
    cloudBGRCSVPath = os.path.join(cloudCSVFolder,cloudBGRCSVName)
    cloudHSVCSVPath = os.path.join(cloudCSVFolder,cloudHSVCSVName)

    skyCSVFolder = r'Sky-CSVS'
    skyBGRCSVName = 'BGRDistribution.csv'
    skyHSVCSVName = "HSVDistribution.csv"
    skyBGRCSVPath = os.path.join(skyCSVFolder,skyBGRCSVName)
    skyHSVCSVPath = os.path.join(skyCSVFolder,skyHSVCSVName)


    cloudPixelsBGR = []
    skyPixelsBGR = []
    cloudPixelsHSV = []
    skyPixelsHSV = []


    #----------------------------------------------------------------------------------------------------#


    """
    We read in our images and convert them to HSV
    """


    blockedImage = cv2.imread(Blocked)
    blockedImageHSV = cv2.cvtColor(blockedImage,cv2.COLOR_BGR2HSV)

    #blockedImage = cv2.resize(blockedImage,(400,300))

    referenceImage = cv2.imread(Reference)
    referenceImageHSV = cv2.cvtColor(referenceImage,cv2.COLOR_BGR2HSV)
    #referenceImage = cv2.resize(referenceImage,(400,300))


    #----------------------------------------------------------------------------------------------------#


    """
    First we make a mask for red colours
    We'll use Red to represent Clouds
    Red can have hue values between 0-10, but also 170-180
    """

    u_b_red1HSV = np.array([10, 255, 255])
    l_b_red1HSV = np.array([0, 30, 30])

    u_b_red2HSV = np.array([180, 255, 255])
    l_b_red2HSV = np.array([170, 50, 50])

    maskOneRedHSV = cv2.inRange(blockedImageHSV,l_b_red1HSV,u_b_red1HSV)
    maskTwoRedHSV = cv2.inRange(blockedImageHSV,l_b_red2HSV,u_b_red2HSV)

    redMaskHSV = cv2.bitwise_or(maskOneRedHSV,maskTwoRedHSV)

    """
    Now we do the same for Black.
    We'll use a range of black to represent The Sky
    """

    u_b_blackHSV = np.array([180, 255,30])
    l_b_blackHSV = np.array([0, 0, 0])

    blackMaskHSV = cv2.inRange(blockedImageHSV,l_b_blackHSV,u_b_blackHSV)

    #-------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------#


    """
    Now we use apply masks via bitwise ands to get our cloud and sky
    """

    cloudImageHSV = cv2.bitwise_and(referenceImageHSV,referenceImageHSV,mask = redMaskHSV)
    skyImageHSV = cv2.bitwise_and(referenceImageHSV,referenceImageHSV,mask = blackMaskHSV)

    cloudImageBGR = cv2.cvtColor(cloudImageHSV,cv2.COLOR_HSV2BGR)
    skyImageBGR =  cv2.cvtColor(skyImageHSV,cv2.COLOR_HSV2BGR)


    #----------------------------------------------------------------------------------------------------#


    """
    We iterate through our two masked images using zip() and take note of the values
    of pixels that aren't true black (0,0,0) by saving them in lists
    """

    #cloudImage = cv2.cvtColor(cloudImage,cv2.COLOR_HSV2BGR)
    #skyImage = cv2.cvtColor(skyImage,cv2.COLOR_HSV2BGR)


    for cloudPixelBGR,skyPixelBGR in zip(cloudImageBGR,skyImageBGR):
         for cloudPixelValueBGR,skyPixelValueBGR in zip(cloudPixelBGR,skyPixelBGR):

            cloudB,cloudG,cloudR = cloudPixelValueBGR
            skyB,skyG,skyR = skyPixelValueBGR

            if cloudB!=0 or cloudG!=0 or cloudR!=0:
                    cloudPixelsBGR.append(list(cloudPixelValueBGR))

            if skyB!=0 or skyG!=0 or skyR!=0:
                    skyPixelsBGR.append(list(skyPixelValueBGR))
            else:
                continue


    for cloudPixel,skyPixel in zip(cloudImageHSV,skyImageHSV):
        for cloudPixelValue,skyPixelValue in zip(cloudPixel,skyPixel):

            cloudH,cloudS,cloudV = cloudPixelValue
            skyH,skyS,skyV = skyPixelValue

            if cloudH!=0 or cloudS!=0 or cloudV!=0:
                    cloudPixelsHSV.append(list(cloudPixelValue))

            if skyH!=0 or skyS!=0 or skyV!=0:
                    skyPixelsHSV.append(list(skyPixelValue))
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
                #print("Writing to csv")
            csvFile.close()

        except Exception as exception:

            DataValueErrorLogName = (datetime.now().strftime("%Y_%b_%d_%H-%M-%S") +".txt")
            DataValueErrorLogPath = (r"ErrorLogs/WriteLogs/"+ DataValueErrorLogName)
            logfile = open(DataValueErrorLogPath,"a")
            errorlog = f"Corrupted or missing value in the form {pixelBGR}"
            logfile.write(errorlog)

            logfile.close()



    """
    Now we create two threads to write to our csv files concurrrently
    """



    skyBGRWriter = threading.Thread(target = writeCsv, args = [skyBGRCSVPath,skyPixelsBGR])
    cloudBGRWriter = threading.Thread(target = writeCsv, args = [cloudBGRCSVPath,cloudPixelsBGR])

    skyHSVWriter = threading.Thread(target = writeCsv, args = [skyHSVCSVPath,skyPixelsHSV])
    cloudHSVWriter = threading.Thread(target = writeCsv, args = [cloudHSVCSVPath,cloudPixelsHSV])

    skyBGRWriter.start()
    cloudBGRWriter.start()
    skyHSVWriter.start()
    cloudHSVWriter.start()

    skyBGRWriter.join()
    cloudBGRWriter.join()
    skyHSVWriter.join()
    cloudHSVWriter.join()


#---------------------------------------------------------------------------------------------------------#

def distributionBarGraphGenerator(cloudCSVFolder,skyCSVFolder,graphFolder,BGRDistributionCSVName,HSVDistributionCSVName,bins):


    """
    We create our Green, Blue, and Red Distributions here.
    """

    BGRCloudDistributionCSVPath = os.path.join(cloudCSVFolder,BGRDistributionCSVName)
    BGRSkyDistributionCSVPath = os.path.join(skyCSVFolder,BGRDistributionCSVName)

    HSVCloudDistributionCSVPath = os.path.join(cloudCSVFolder,HSVDistributionCSVName)
    HSVSkyDistributionCSVPath = os.path.join(skyCSVFolder,HSVDistributionCSVName)

    bgrGraphsavePath = os.path.join(graphFolder,'BGRBarGraph.pdf')
    hsvGraphsavePath = os.path.join(graphFolder,'HSVBarGraph.pdf')


    def readDistributionData(distributionCsv):
        blues = []
        greens = []
        reds = []

        with open(distributionCsv,'r') as file:

            print("FETCHING DATAPOINTS.")

            DataValueErrorLogName = (datetime.now().strftime("%Y_%b_%d_%H-%M-%S")+".txt")
            DataValueErrorLogNamePath = r"ErrorLogs/ReadLogs/"+ DataValueErrorLogName
            logfile = open(DataValueErrorLogNamePath,'a+')

            csvReader = csv.reader(file)

            for row in csvReader:
                try:
                    blues.append(int(row[0]))
                    greens.append(int(row[1]))
                    reds.append(int(row[2]))

                    #if ("BGR" in distributionCsv) and ("Cloud" in distributionCsv):
                        #print(f"Reading {row} from {distributionCsv} ")

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

        BGRCloudDistributionDataThread = executor.submit(readDistributionData,BGRCloudDistributionCSVPath)
        BGRSkyDistributionDataThread = executor.submit(readDistributionData,BGRSkyDistributionCSVPath)
        HSVCloudDistributionDataThread = executor.submit(readDistributionData,HSVCloudDistributionCSVPath)
        HSVSkyDistributionDataThread = executor.submit(readDistributionData,HSVSkyDistributionCSVPath)

        BGRCloudDistribution = BGRCloudDistributionDataThread.result()
        BGRSkyDistribution = BGRSkyDistributionDataThread.result()
        HSVCloudDistribution = HSVCloudDistributionDataThread.result()
        HSVSkyDistribution = HSVSkyDistributionDataThread.result()


    #print(len(BGRCloudDistribution))


    cloudBlues,cloudGreens,cloudReds = BGRCloudDistribution
    skyBlues,skyGreens,skyReds = BGRSkyDistribution

    skyHues, skySats, skyValues = HSVSkyDistribution
    cloudHues, cloudSats, cloudValues = HSVCloudDistribution

    print(f"- There are: \n > {len(cloudBlues)} Blue cloud datapoints,\n > {len(cloudGreens)} Green cloud datapoints \n > {len(cloudReds)} Red cloud datapoints")

    print("\n> CREATING BGR GRAPH...")
    fig1,axes1 = plt.subplots(nrows = 3,ncols = 1)
    axes1 = axes1.flatten()

    axes1[0].hist(skyBlues, bins = bins,color = 'blue',alpha= 0.3,label = 'Sky Blues')
    axes1[0].hist(cloudBlues,bins = bins,color = 'purple',alpha = 0.3,label = 'Cloud Blues')
    axes1[0].set_xlabel('BGR Blues (0 - 255)')
    axes1[0].set_ylabel('frequency')
    axes1[0].legend(loc="upper left")

    axes1[1].hist(cloudGreens, bins = bins,color = 'green',alpha= 0.3,label = 'Cloud Greens')
    axes1[1].hist(skyGreens,bins = bins,color = 'yellow',alpha = 0.3,label = 'Sky Greens')
    axes1[1].set_xlabel('BGR Greens (0 - 255)')
    axes1[1].set_ylabel('frequency')
    axes1[1].legend(loc="upper left")

    axes1[2].hist(cloudReds, bins = bins,color = 'red',alpha= 0.3,label = 'Cloud Reds')
    axes1[2].hist(skyReds,bins = bins,color = 'pink',alpha = 0.3,label = 'Sky Reds')
    axes1[2].set_xlabel('BGR Reds(0 - 255)')
    axes1[2].set_ylabel('frequency')
    axes1[2].legend(loc="upper left")

    fig1.tight_layout()
    plt.savefig(bgrGraphsavePath)
    fig1.clear()
    plt.close(fig1)


    print(" \n> CREATING HSV GRAPH ...")
    fig2,axes2 = plt.subplots(nrows = 3,ncols = 1)
    axes2 = axes2.flatten()

    axes2[0].hist(skyHues, bins = bins,color = 'blue',alpha= 0.3,label = 'Sky Hues')
    axes2[0].hist(cloudHues,bins = bins,color = 'purple',alpha = 0.3,label = 'Cloud Hues')
    axes2[0].set_xlabel('HSV Hues (0 - 255)')
    axes2[0].set_ylabel('frequency')
    axes2[0].legend(loc="upper left")

    axes2[1].hist(cloudValues, bins = bins,color = 'green',alpha= 0.3,label = 'Cloud Saturation')
    axes2[1].hist(skyValues,bins = bins,color = 'yellow',alpha = 0.3,label = 'Sky Saturation')
    axes2[1].set_xlabel('HSV Saturation (0 - 255)')
    axes2[1].set_ylabel('frequency')
    axes2[1].legend(loc="upper left")

    axes2[2].hist(cloudSats, bins = bins,color = 'red',alpha= 0.3,label = 'Cloud Value')
    axes2[2].hist(skySats,bins = bins,color = 'pink',alpha = 0.3,label = 'Sky Value')
    axes2[2].set_xlabel('HSV Value (0 - 255)')
    axes2[2].set_ylabel('frequency')
    axes2[2].legend(loc="upper left")

    fig2.tight_layout()
    plt.savefig(hsvGraphsavePath)
    fig2.clear()
    plt.close(fig2)






#---------------------------------------------------------------------------------------------------------#


"""
This is our main function area.
"""

def main():

    start = datetime.now()
    """
    Space for important variables
    """
    bins = [*range(0,256,1)]
    FuckedImagesCounter = 0
    blockedImageFolder = r'Blocked-Images'
    #blockedImageName = r''
    #blockedImagePath = os.path.join(blockedImageFolder,blockedImageName)

    referenceImageFolder = r'Reference-Images'
    #referenceImageName = r''
    #referenceImagePath = os.path.join(referenceImageFolder,referenceImageName)


    activeProcesses = []

    imagePairList=[]

    counter = 0


    cloudCSVFolder = r'Cloud-CSVS'
    cloudBGRCSVName = 'BGRDistribution.csv'
    cloudHSVCSVName="HSVDistribution.csv"
    cloudBGRCSVPath = os.path.join(cloudCSVFolder,cloudBGRCSVName)
    cloudHSVCSVPath = os.path.join(cloudCSVFolder,cloudHSVCSVName)

    skyCSVFolder = r'Sky-CSVS'
    skyBGRCSVName = 'BGRDistribution.csv'
    skyHSVCSVName = "HSVDistribution.csv"
    skyBGRCSVPath = os.path.join(skyCSVFolder,skyBGRCSVName)
    skyHSVCSVPath = os.path.join(cloudCSVFolder,skyHSVCSVName)

    graphFolder = r"Graphs"

    BGRDistributionCSVName = "BGRDistribution.csv"

    HSVDistributionCSVName = "HSVDistribution.csv"



#---------------------------------------------------------------------------------------------------------#
    """
    If our distribution csvs already exist, we need to delete them
    """


    if os.path.isfile(cloudBGRCSVPath):
        os.remove(cloudBGRCSVPath)

    if os.path.isfile(skyBGRCSVPath):
        os.remove(skyBGRCSVPath)

    if os.path.isfile(cloudHSVCSVPath):
        os.remove(cloudHSVCSVPath)

    if os.path.isfile(skyHSVCSVPath):
        os.remove(skyHSVCSVPath)

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
    distributionBarGraphGenerator(cloudCSVFolder,skyCSVFolder,graphFolder,BGRDistributionCSVName,HSVDistributionCSVName,bins)
#---------------------------------------------------------------------------------------------------------#

    runtimeDelta = datetime.now()

    runtime  = runtimeDelta - start

    print(f'Runtime : {runtime}')

#---------------------------------------------------------------------------------------------------------#


if __name__ == '__main__':
    main()
