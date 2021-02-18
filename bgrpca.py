__author__ = 'Tadj Cazaubon'
__credits__ = ["Tadj Cazaubon",'Tohid Ardeshiri']


import os
import csv
import pandas as pd
import numpy as np
import pickle
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

def pixelCheck(skyReds,cloudReds):

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

    print(f'\n > There are : \n {len(skyPixelLabel)} sky pixels \n {len(cloudPixelLabel)} cloud pixels ')

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
    global saveVar

    if os.path.exists(savePathDF):
        saveVar = True
        print('\n> Existing Dataframe detected. Attempting to import . . .')

        try:
            dataframe = pd.read_pickle(savePathDF)
            #+print(dataframe.head())
            print("\n > Cache data imported successfully.")

        except Exception as e:
            
            print('File is either open or corrupted. Please close or delete it.')
            saveVar=False



    else:
        saveVar=False

        print('\n> No Dataframe detected. Creating DataFrame . . .')

        dataFrameTimerStart = datetime.now()

        dataframe = pd.DataFrame(columns=[*cloudPixelLabel,*skyPixelLabel],index = labels)

        print('\n> DataFrame created in ' + str(datetime.now()-dataFrameTimerStart))

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

        print(f'\n> Imported BGR Dataframe not valid for current dataset. Attempting to create new Dataframe . . . \n')

        dataFrameTimerStart = datetime.now()

        dataframe = pd.DataFrame(columns=[*cloudPixelLabel,*skyPixelLabel],index = labels)

        print('\n> DataFrame created in ' + str(datetime.now()-dataFrameTimerStart))

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


def fillHSVDataframe(dataframe,skyPixelLabel,cloudPixelLabel,cloudRGBList,skyRGBList,savePathDF,labels):

    """
    Now we fill our dataframe.
    """


    finalSkyPixelIndex = ('skyPixel' + str(len(skyPixelLabel)))
    finalCloudPixelIndex = ('cloudPixel' + str(len(cloudPixelLabel)))

    #print(dataframe.head())


    try:
        #print(f'\n > Now filling data for {color} . . .')
        dataframe = dataframe.T
        dataframe.loc['cloudPixel1':finalCloudPixelIndex,'values':'hues'] = cloudRGBList
        dataframe.loc['skyPixel1':finalSkyPixelIndex,'values':'hues'] = skyRGBList



        print('\n')

        return (dataframe,finalCloudPixelIndex,finalSkyPixelIndex)

    except Exception as e:

        print(f'\n> Imported HSV Dataframe not valid for current dataset. Attempting to create new Dataframe . . . \n')

        dataFrameTimerStart = datetime.now()

        dataframe = pd.DataFrame(columns=[*cloudPixelLabel,*skyPixelLabel],index = labels)

        print('\n> DataFrame created in ' + str(datetime.now()-dataFrameTimerStart))

        try:
            dataframe.to_pickle(savePathDF)
            print(dataframe.head())

        except Exception as e:
            print(f'\n {e} \n')

        dataframe = dataframe.T
        dataframe.loc['cloudPixel1':finalCloudPixelIndex,'values':'hues'] = cloudRGBList
        dataframe.loc['skyPixel1':finalSkyPixelIndex,'hues':'values'] = skyRGBList



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


    print('\n> Creating final PCA DataFrame ...')
    dataFrameTimerStart2 = datetime.now()
    pcaDataframe = pd.DataFrame(pcaData*10,index=[*cloudPixelLabel,*skyPixelLabel],columns=pcaLabels)
    print('\n> DataFrame Created in ' + str(datetime.now()-dataFrameTimerStart2))
    print("---------------------------------------------------------")
    print(pcaDataframe.head())
    print("---------------------------------------------------------")


    """
    Now we separate our one dataframe into two, one for the sky and one for the
    clouds. This is so we can colour-code our data points.
    """

    cloudPcaDataframe = pcaDataframe.loc['cloudPixel1':finalCloudPixelIndex]


    skyPcaDataframe = pcaDataframe.loc['skyPixel1':finalSkyPixelIndex]

    #print(skyPcaDataframe.head())

    return (cloudPcaDataframe,skyPcaDataframe)


def createBGRScatter(cloudPcaDataframe,skyPcaDataframe,varPercent,savePathPCA):

    """
    Now we can create a scatterplot of our data
    """

    if saveVar == True:

        print("\n> Data is identical to last known run. Checking for cached BGR scatterplot ...")

        if os.path.exists(savePathPCA):

            print("\n> Cached BGR scatterplot found. Checking cached BGR data compatibility ...")

            try:
                scatterPlotTimerStart = datetime.now()
                fig = pickle.load(open(savePathPCA,'rb'))
                print("\n> Cached BGR scatterplot successfully imported")
                plt.show()

            except Exception as e:
                print( "\n> Cached BGR data could not be read/loaded.")
        else:
            print('\n Cached BGR scatterplot data not found, creating BGR ScatterPlot ...')
            scatterPlotTimerStart = datetime.now()

            fig,ax = plt.subplots(figsize=(10,6))
            ax.scatter(cloudPcaDataframe.PC1,cloudPcaDataframe.PC2, c = 'lightblue',alpha = 0.4,marker = 'X',label = 'Cloud Value')
            ax.scatter(skyPcaDataframe.PC1,skyPcaDataframe.PC2,c = 'red',alpha = 0.1,marker = 'o',label = 'Sky Value')
            plt.legend(loc="upper left")

            plt.title('RGB PCA GRAPH')
            plt.xlabel(f'PC1 RED {varPercent[0]}%')
            plt.ylabel(f'PC2 GREEN {varPercent[1]}%')
            #plt.tight_layout()

            """it may be a good idea to make showing and saving the graphs two separate processes."""
            #plt.show()
            try:
                pickle.dump(fig,open(savePathPCA,'wb'))
                print("\n> BGR scatterplot successfully saved.")
            except Exception as e:
                print("\n> BGR scatterplot could not be saved correctly. Data will not be cached.")
            plt.close("all")
    else:
        print('\n Cached BGR scatterplot data not available, creating BGR ScatterPlot ...')
        scatterPlotTimerStart = datetime.now()

        fig,ax = plt.subplots(figsize=(10,6))
        ax.scatter(cloudPcaDataframe.PC1,cloudPcaDataframe.PC2, c = 'lightblue',alpha = 0.4,marker = 'X',label = 'Cloud Value')
        ax.scatter(skyPcaDataframe.PC1,skyPcaDataframe.PC2,c = 'red',alpha = 0.1,marker = 'o',label = 'Sky Value')
        plt.legend(loc="upper left")

        plt.title('RGB PCA GRAPH')
        plt.xlabel(f'PC1 RED {varPercent[0]}%')
        plt.ylabel(f'PC2 GREEN {varPercent[1]}%')
        #plt.tight_layout()

        """it may be a good idea to make showing and saving the graphs two separate processes."""
        #plt.show()
        try:
            pickle.dump(fig,open(savePathPCA,'wb'))
            print("\n> BGR scatterplot successfully saved.")
        except Exception as e:
             print("\n> BGR scatterplot could not be saved correctly. Data will not be cached.")
        plt.close("all")


    #print('\n ScatterPlot Created in ' + str(datetime.now()-scatterPlotTimerStart))


def createHSVScatter(cloudPcaDataframe,skyPcaDataframe,varPercent,savePathPCA):

    """
    Now we can create a scatterplot of our data
    """
   # print(f"SaveVar is currently {saveVar}")
    if saveVar == True:
        
        print("\n> Data is identical to last known run. Checking for cached HSV scatterplot ...")

        if os.path.exists(savePathPCA):

            print("\n> Cached HSV scatterplot found. Checking cached HSV data compatibility ...")

            try:
                scatterPlotTimerStart = datetime.now()
                fig = pickle.load(open(savePathPCA,'rb'))
                print("\n> Cached HSV scatterplot successfully imported.")
                plt.show()

            except Exception as e:
                print( "\n> Cached HSV data could not be read/loaded.")
        else:
            print("\n> Cached HSV scatterplot not found. ")
            scatterPlotTimerStart = datetime.now()

            fig,ax = plt.subplots(figsize=(10,6))
            ax.scatter(cloudPcaDataframe.PC1,cloudPcaDataframe.PC2, c = 'lightblue',alpha = 0.4,marker = 'X',label = 'Cloud Value')
            ax.scatter(skyPcaDataframe.PC1,skyPcaDataframe.PC2,c = 'red',alpha = 0.1,marker = 'o',label = 'Sky Value')
            plt.legend(loc="upper left")

            plt.title('HSV PCA GRAPH')
            plt.xlabel(f'PC1 VALUE {varPercent[0]}%')
            plt.ylabel(f'PC2 SATURATION {varPercent[1]}%')
            #plt.tight_layout()

            """it may be a good idea to make showing and saving the graphs two separate processes."""
            #plt.show()

            try:
                pickle.dump(fig,open(savePathPCA,'wb'))
                print("\n> HSV scatterplot successfully saved. ")
            except Exception as e:
                print("\n> HSV PCA scatterplot could not be saved correctly. Data will not be cached.")

            plt.close("all")
    else:
        print('\n Cached data not available, creating BGR ScatterPlot ...')
    
        scatterPlotTimerStart = datetime.now()

        fig,ax = plt.subplots(figsize=(10,6))
        ax.scatter(cloudPcaDataframe.PC1,cloudPcaDataframe.PC2, c = 'lightblue',alpha = 0.4,marker = 'X',label = 'Cloud Value')
        ax.scatter(skyPcaDataframe.PC1,skyPcaDataframe.PC2,c = 'red',alpha = 0.1,marker = 'o',label = 'Sky Value')
        plt.legend(loc="upper left")

        plt.title('HSV PCA GRAPH')
        plt.xlabel(f'PC1 VALUE {varPercent[0]}%')
        plt.ylabel(f'PC2 SATURATION {varPercent[1]}%')
        #plt.tight_layout()

        """it may be a good idea to make showing and saving the graphs two separate processes."""
        #plt.show()

        try:
            pickle.dump(fig,open(savePathPCA,'wb'))
            print("\n> HSV scatterplot successfully saved. ")
        except Exception as e:
            print("\n> HSV PCA scatterplot could not be saved correctly. Data will not be cached.")

        plt.close("all")

    #print('\n ScatterPlot Created in ' + str(datetime.now()-scatterPlotTimerStart))


#---------------------------------------------------------------------------------------------------------#

def main():


    """
    Set data paths and start runtime timer.
    """

    start = datetime.now()


    cloudBGRDistribution = r'Cloud-CSVS/BGRDistribution.csv'
    skyBGRDistribution = r'Sky-CSVS/BGRDistribution.csv'
    cloudHSVDistribution = r'Cloud-CSVS/HSVDistribution.csv'
    skyHSVDistribution = r'Sky-CSVS/HSVDistribution.csv'

    savePathBGRScree = r'Graphs/BGRScreePlot.pdf'
    savePathHSVScree = r'Graphs/HSVScreePlot.pdf'

    savePathPCABGR = r'Graphs/BGRPCAPlot.pkl'
    savePathPCAHSV = r'Graphs/HSVPCAPlot.pkl'

    savePathBGRDF = r'Graphs/BGRDataFrame.pkl'
    savePathHSVDF = r'Graphs/HSVDataFrame.pkl'


    #-----------------------------------------------------------------------------------------------------#

    cloudReds=[]
    cloudGreens = []
    cloudBlues = []

    skyReds = []
    skyGreens = []
    skyBlues = []


    cloudHues=[]
    cloudSats=[]
    cloudValues=[]

    skyHues=[]
    skySats=[]
    skyValues=[]


    BGRlabels=['red','green','blue']
    HSVlabels=["values","sats","hues"]

    #---------------------------------------------------------------------------------------------------------#

    """
    Pull sky and cloud colour data from csv files by passing path.
    """

    cloudRGBList = dataPull(cloudBGRDistribution)
    skyRGBList = dataPull(skyBGRDistribution)

    skyHSVList = dataPull(skyHSVDistribution)
    cloudHSVList = dataPull(cloudHSVDistribution)



    for RGBValue in cloudRGBList:
        cloudReds.append(RGBValue[0])
        cloudGreens.append(RGBValue[1])
        cloudBlues.append(RGBValue[2])

    for RGBValue in skyRGBList:
        skyReds.append(RGBValue[0])
        skyGreens.append(RGBValue[1])
        skyBlues.append(RGBValue[2])


    """
    HSV values are flipped because I read the BGR values in the order RBG in the datapull function.
    """
    for HSVValue in cloudHSVList:
        cloudHues.append(HSVValue[2])
        cloudSats.append(HSVValue[1])
        cloudValues.append(HSVValue[0])

    for HSVValue in skyHSVList:
        skyHues.append(HSVValue[2])
        skySats.append(HSVValue[1])
        skyValues.append(HSVValue[0])

#---------------------------------------------------------------------------------------------------------#

    """
    Create Column labels for future DataFrame.

    Also check ensure the number of cloud and sky pixels match up
    with the data passed.
    """

    BGRcloudPixelLabel,BGRskyPixelLabel = pixelCheck(skyReds,cloudReds)
    HSVcloudPixelLabel,HSVskyPixelLabel = pixelCheck(skyHues,cloudHues)


#---------------------------------------------------------------------------------------------------------#
    """
    With the first run, Dataframe creation took an average of about 4 minutes.
    to fix this, I've decided to save future runs as .pkl files.

    Firstly we check whether the dataframe exists as a file with the
    extension pkl.
    If no dataframe exists we create one.
    """

    BGRdataframe = dataframeFrame(savePathBGRDF,BGRcloudPixelLabel,BGRskyPixelLabel,BGRlabels)
    HSVdataframe = dataframeFrame(savePathHSVDF,HSVcloudPixelLabel,HSVskyPixelLabel,HSVlabels)

    #print(dataframe.head())

    BGRdataframeFilled,finalBGRCloudPixelIndex,finalBGRSkyPixelIndex = fillDataframe(BGRdataframe,BGRskyPixelLabel,BGRcloudPixelLabel,cloudRGBList,skyRGBList,savePathBGRDF,BGRlabels)
    HSVdataframeFilled,finalHSVCloudPixelIndex,finalHSVSkyPixelIndex = fillHSVDataframe(HSVdataframe,HSVskyPixelLabel,HSVcloudPixelLabel,cloudHSVList,skyHSVList,savePathHSVDF,HSVlabels)

#---------------------------------------------------------------------------------------------------------#

    """
    Now we preprocess,scale, fit, and analyze or dataframe data with sklearn
    """

    BGRpcaData,BGRvarPercent = processData(BGRdataframeFilled)
    HSVpcaData,HSVvarPercent = processData(HSVdataframeFilled)

#---------------------------------------------------------------------------------------------------------#

    """
    Now we create our Bar Graph
    """

    createBarGraph(BGRvarPercent,BGRlabels,savePathBGRScree)
    createBarGraph(HSVvarPercent,HSVlabels,savePathHSVScree)

#---------------------------------------------------------------------------------------------------------#

    """
    Now that we know PC1 and PC2, we can construct a dataframe to make a PCA plot of them.

    Now we separate our one dataframe into two, one for the sky and one for the
    clouds. This is so we can colour-code our data points.
    """

    BGRcloudPcaDataframe,BGRskyPcaDataframe = createPcaDataframe(BGRvarPercent,BGRpcaData,BGRcloudPixelLabel,BGRskyPixelLabel,finalBGRCloudPixelIndex,finalBGRSkyPixelIndex)
    HSVcloudPcaDataframe,HSVskyPcaDataframe = createPcaDataframe(HSVvarPercent,HSVpcaData,HSVcloudPixelLabel,HSVskyPixelLabel,finalHSVCloudPixelIndex,finalHSVSkyPixelIndex)

#---------------------------------------------------------------------------------------------------------#

    """
    Now we can create a scatterplot of our data
    """

    createBGRScatter(BGRcloudPcaDataframe,BGRskyPcaDataframe,BGRvarPercent,savePathPCABGR)
    createHSVScatter(HSVcloudPcaDataframe,HSVskyPcaDataframe,HSVvarPercent,savePathPCAHSV)

#---------------------------------------------------------------------------------------------------------#

    runtimeDelta = datetime.now()

    runtime  = runtimeDelta - start

    print(f'Runtime : {runtime}')

#---------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    main()
