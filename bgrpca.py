__author__ = 'Tadj Cazaubon'
__credits__ = ["Tadj Cazaubon",'Tohid Ardeshiri']


import os
import csv
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from datetime import datetime
from matplotlib import pyplot as plt


#---------------------------------------------------------------------------------------------------------#


"""
Pull sky and cloud colour data from csv files by passing path.
"""


def dataPull(distributionPath):

    blues = []
    reds = []
    greens = []
    RGBValueList = []
    with open(distributionPath,'r') as File:

        csvReader = csv.reader(File)

        for row in csvReader:

            try:

                RGBValueList.append([int(row[2]),int(row[1]),int(row[0])])
            except Exception as e:
                print(f'\n Data missing or corrupted at line {(len(RGBValueList)+1)} in {distributionPath}')

    File.close()


    print(f' \n Values are in the form R: {RGBValueList[0][0]} , G: {RGBValueList[0][1]} , B: {RGBValueList[0][2]} with a type of {type(RGBValueList[0][0])}')

    return (RGBValueList)


#---------------------------------------------------------------------------------------------------------#

"""
Create Column labels for future DataFrame.

Also check ensure the number of cloud and sky pixels match up
with the data passed.
"""

def cloudCheck(skyReds,cloudReds):

    skyPixelLabel = []
    cloudPixelLabel = []

    print('\n Counting Sky & Cloud Pixels . . .')

    count = 1
    for skyPixel in skyReds:
        skyPixelLabel.append('skyPixel'+str(count))
        count+=1

    count = 1
    for cloudPixel in cloudReds:
        cloudPixelLabel.append('cloudPixel'+str(count))
        count+=1

    print(f'\n There are : \n {len(skyPixelLabel)} sky pixels \n {len(cloudPixelLabel)} cloud pixels')

    return(cloudPixelLabel,skyPixelLabel)

#---------------------------------------------------------------------------------------------------------#

def  dataframeFrame(savePathDF,cloudPixelLabel,skyPixelLabel,labels):
    """
    With the first run, Dataframe creation took an average of about 4 minutes.
    to fix this, I've decided to save future runs as .pkl files.

    Firstly we check whether the dataframe exists as a file with the
    extension pkl.
    If no dataframe exists we create one.
    """

    if os.path.exists(savePathDF):

        print('\n Existing Dataframe detected. Attempting to import . . .')

        try:
            dataframe = pd.read_pickle(savePathDF)
            #+print(dataframe.head())

        except Exception as e:

            print('File is either open or corrupted. Please close or delete it.')



    else:

        print('\n No Dataframe detected. Creating DataFrame . . .')

        dataFrameTimerStart = datetime.now()

        dataframe = pd.DataFrame(columns=[*cloudPixelLabel,*skyPixelLabel],index = labels)

        print('\n DataFrame created in ' + str(datetime.now()-dataFrameTimerStart))

        try:
            dataframe.to_pickle(savePathDF)
            print(dataframe.head())

        except Exception as e:
            print(e)

    return dataframe



"""
We need to check our dataframe dimensions and compare it to our  image pixel data.
If we attempt to fit our data to our dataframe and an <insert error type> is
raised, we need to recreate our dataframe template to fit the dimensions.
"""




def fillDataframe(dataframe,skyPixelLabel,cloudPixelLabel,cloudRGBList,skyRGBList,savePathDF,labels):

    """
    Now we fill our dataframe.
    """


    finalSkyPixelIndex = ('skyPixel' + str(len(skyPixelLabel)))
    finalCloudPixelIndex = ('cloudPixel' + str(len(cloudPixelLabel)))

    #print(dataframe.head())


    try:
        #print(f'\n > Now filling data for {color} . . .')
        dataframe = dataframe.T
        dataframe.loc['cloudPixel1':finalCloudPixelIndex,'red':'blue'] = cloudRGBList
        dataframe.loc['skyPixel1':finalSkyPixelIndex,'red':'blue'] = skyRGBList



        print('\n')

        return (dataframe,finalCloudPixelIndex,finalSkyPixelIndex)

    except Exception as e:

        print(f'\n Imported Dataframe not valid for current dataset. Attempting to create new Dataframe . . . \n')

        dataFrameTimerStart = datetime.now()

        dataframe = pd.DataFrame(columns=[*cloudPixelLabel,*skyPixelLabel],index = labels)

        print('\n DataFrame created in ' + str(datetime.now()-dataFrameTimerStart))

        try:
            dataframe.to_pickle(savePathDF)
            print(dataframe.head())

        except Exception as e:
            print(f'\n {e} \n')

        dataframe = dataframe.T
        dataframe.loc['cloudPixel1':finalCloudPixelIndex,'red':'blue'] = cloudRGBList
        dataframe.loc['skyPixel1':finalSkyPixelIndex,'red':'blue'] = skyRGBList



        print('\n')

        return (dataframe,finalCloudPixelIndex,finalSkyPixelIndex)

#---------------------------------------------------------------------------------------------------------#

def processData(dataframe):

    """
    Now we preprocess,scale, fit, and analyze or dataframe data with sklearn
    """

    scaledData = preprocessing.scale(dataframe)

    pca = PCA()

    pca.fit(scaledData)
    pcaData = pca.transform(scaledData)

    varPercent = np.round(pca.explained_variance_ratio_*100,decimals=1)

    return(pcaData,varPercent)

#---------------------------------------------------------------------------------------------------------#


def createBarGraph(varPercent,labels,savePathScree):

    """
    Now we make a Scree Plot to visualize each PC's variance percentage.
    We just use a bar graph and scale the heights to our variance percentages
    """

    plt.bar(x=range(1,len(varPercent)+1),height = varPercent,tick_label=labels)

    plt.ylabel('Variance Percentage')
    plt.xlabel('Principle Component')
    plt.title('Scree Plot')
    plt.savefig(savePathScree)
    plt.clf()

#---------------------------------------------------------------------------------------------------------#


def createPcaDataframe(varPercent,pcaData,cloudPixelLabel,skyPixelLabel,finalCloudPixelIndex,finalSkyPixelIndex):

    """
    Now that we know PC1 and PC2, we can construct a dataframe to make a PCA plot of them.
    """


    pcaLabels = ['PC' + str(x) for x in range(1,len(varPercent)+1)]


    print('\n Creating DataFrame.')
    dataFrameTimerStart2 = datetime.now()
    pcaDataframe = pd.DataFrame(pcaData*10,index=[*cloudPixelLabel,*skyPixelLabel],columns=pcaLabels)
    print('\n DataFrame Created in ' + str(datetime.now()-dataFrameTimerStart2))

    print(pcaDataframe.head())


    """
    Now we separate our one dataframe into two, one for the sky and one for the
    clouds. This is so we can colour-code our data points.
    """

    cloudPcaDataframe = pcaDataframe.loc['cloudPixel1':finalCloudPixelIndex]


    skyPcaDataframe = pcaDataframe.loc['skyPixel1':finalSkyPixelIndex]

    #print(skyPcaDataframe.head())

    return (cloudPcaDataframe,skyPcaDataframe)


def createScatter(cloudPcaDataframe,skyPcaDataframe,varPercent,savePathPCA):

    """
    Now we can create a scatterplot of our data
    """


    print('\n Creating ScatterPlot.')
    scatterPlotTimerStart = datetime.now()

    plt.scatter(cloudPcaDataframe.PC1,cloudPcaDataframe.PC2, c = 'lightblue',alpha = 0.4,marker = 'X',label = 'Cloud Value')
    plt.scatter(skyPcaDataframe.PC1,skyPcaDataframe.PC2,c = 'red',alpha = 0.1,marker = 'o',label = 'Sky Value')
    plt.legend(loc="upper left")

    plt.title('RGB PCA GRAPH')
    plt.xlabel(f'PC1 RED {varPercent[0]}%')
    plt.ylabel(f'PC2 GREEN {varPercent[1]}%')
    #plt.tight_layout()

    """it may be a good idea to make showing and saving the graphs two separate processes."""
    plt.show()
    plt.savefig(savePathPCA)


    print('\n ScatterPlot Created in ' + str(datetime.now()-scatterPlotTimerStart))


#---------------------------------------------------------------------------------------------------------#

def main():


    """
    Set data paths and start runtime timer.
    """

    start = datetime.now()


    cloudDistribution = r'Main\Cloud-CSVS\BGRDistribution.csv'
    skyDistribution = r'Main\Sky-CSVS\BGRDistribution.csv'
    savePathScree = r'Main\Graphs\ScreePlot.pdf'
    savePathPCA = r'Main\Graphs\PCAPlot.pdf'
    savePathDF = r'Main\Graphs\templateDataFrame.pkl'


    cloudReds=[]
    cloudGreens = []
    cloudBlues = []

    skyReds = []
    skyGreens = []
    skyBlues = []


    labels=['red','green','blue']

#---------------------------------------------------------------------------------------------------------#

    """
    Pull sky and cloud colour data from csv files by passing path.
    """

    cloudRGBList = dataPull(cloudDistribution)
    skyRGBList = dataPull(skyDistribution)


    for RGBValue in cloudRGBList:
        cloudReds.append(RGBValue[0])
        cloudGreens.append(RGBValue[1])
        cloudBlues.append(RGBValue[2])


    for RGBValue in skyRGBList:
        skyReds.append(RGBValue[0])
        skyGreens.append(RGBValue[1])
        skyBlues.append(RGBValue[2])

#---------------------------------------------------------------------------------------------------------#

    """
    Create Column labels for future DataFrame.

    Also check ensure the number of cloud and sky pixels match up
    with the data passed.
    """

    cloudPixelLabel,skyPixelLabel = cloudCheck(skyReds,cloudReds)

#---------------------------------------------------------------------------------------------------------#
    """
    With the first run, Dataframe creation took an average of about 4 minutes.
    to fix this, I've decided to save future runs as .pkl files.

    Firstly we check whether the dataframe exists as a file with the
    extension pkl.
    If no dataframe exists we create one.
    """

    dataframe = dataframeFrame(savePathDF,cloudPixelLabel,skyPixelLabel,labels)

    #print(dataframe.head())

    dataframeFilled,finalCloudPixelIndex,finalSkyPixelIndex = fillDataframe(dataframe,skyPixelLabel,cloudPixelLabel,cloudRGBList,skyRGBList,savePathDF,labels)

#---------------------------------------------------------------------------------------------------------#

    """
    Now we preprocess,scale, fit, and analyze or dataframe data with sklearn
    """

    pcaData,varPercent = processData(dataframeFilled)
#---------------------------------------------------------------------------------------------------------#

    """
    Now we create our Bar Graph
    """

    createBarGraph(varPercent,labels,savePathScree)
#---------------------------------------------------------------------------------------------------------#

    """
    Now that we know PC1 and PC2, we can construct a dataframe to make a PCA plot of them.

    Now we separate our one dataframe into two, one for the sky and one for the
    clouds. This is so we can colour-code our data points.
    """

    cloudPcaDataframe,skyPcaDataframe = createPcaDataframe(varPercent,pcaData,cloudPixelLabel,skyPixelLabel,finalCloudPixelIndex,finalSkyPixelIndex)

#---------------------------------------------------------------------------------------------------------#

    """
    Now we can create a scatterplot of our data
    """

    createScatter(cloudPcaDataframe,skyPcaDataframe,varPercent,savePathPCA)

#---------------------------------------------------------------------------------------------------------#

    runtimeDelta = datetime.now()

    runtime  = runtimeDelta - start

    print(f'Runtime : {runtime}')

#---------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    main()
