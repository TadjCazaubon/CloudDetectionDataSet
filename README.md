# CloudDetectionDataSet


# NOTE: Likely this Project will never be completed due to personal reasons 

This is my cloud detection project, done for/with Tohid Ardeshiri.

It was made to show the principle components of sky and cloud BGR values to aid in more accurate real-time detection of clouds. 

As of now,'createCsvs.py' takes a given sample of images of the sky in the folder 'References-Images'. Their same named counterparts are in the folder 'Blocked-Images'.
  
  -Each image in 'Blocked-Images' is color-blocked, using red to label the clouds, and black to label the sky. The two image     
  folders are iterated over using a zip() and Python3's os.walk(). Each pixel corresponding to one color-blocked as either 
  cloud or sky is saved in a list of sky or cloud pixel BGR values.
  
    -This is significantly sped up using the multiprocessing and concurrent.futures modules in Python3. Each iteration of the 
    images is opened as a separate process. I have not capped the number of processes as of yet, but it possible in the future.
    This enables more scalability for larger datasets and to run on more powerful hardware.
  
  
  -The sky and cloud pixel BGR value arrays are written to csv files in Sky-CSVS and Cloud-CSVS respectively. The are both named   
  'BGRDistribution.csv'.This step is sped up a bit using the threading moduleof Python3. Threading chosen over Multi-Processing 
  because the operation is IO-Bound. The values are written to the files as a method of debugging/investigating lost/corrupted 
  data values.
  
  
  -These BGR values are graphed on a Histogram via matplotlib, as to have the BGR values distributions contrasted for the sky and 
  clouds. Each color channel has its own subplot, with the two distributions show with a low enough alpha to see any points of 
  intersection.
  

'bgrpca.py' reads our BGR data from our respective BGR csvs and creates a dataframe template based on the size of our data.

  -A later addition was saving dataframe templates using .pkl and implementing error-handling when the saved dataframe didn't fit  
  the new data sizes on runs with an updated dataset. If the template does not match the data, a new one is created and the   
  saved one is overwritten.
  
  -The BGR data is then split into the three color channels and mapped onto our dataframe, with the colours as indexes and each pixel  
  as a column.
  
  -The Dataframe is preprocessed, scaled, fit and transformed using the sklearn library.
  
    -This is where we get our pca Data(loading scores) to be fit to our graph, and also the variance percent of each principle 
    component.
    
    -A new PCA dataframe is created to later graph our results, with our principle components as labels, and our pixels as columns.
      
      -This dataframe is split into two separate dataframes, one for our sky pixels, and one for clouds.
      These are then mapped onto a scatterplot, giving each dataframe a colour on our graph to better visualise our data.
      Our axes are the two highest components, PC1(red), and PC2(green).
      
        -The variance percventage and therefore thre margin by which these channels are higher than blue varies with the dataset.
    
