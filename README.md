
# KNN implementation

## Python
For me to implement the K Nearest Neighbour algorithms, I had to convert the features data points that contained categorical data to numerical. The new data set was then saved onto a new CSV file called data.new.csv. I imported Panda Library to import my CSV datasets The CSV data was then converted into a list whereby I was able to split the list into 80% Test Data and 20% Train Data. The two classes were classified as 1; being over 50K, and 0; which represented being under 50K. After this, I created a function called K Neared Neighbour, which takes three parameters; dataset, the prediction and the value of K. I than converted my list into an array using the library Numpy array, once this was done, I used the Euclidean distance formula to calculate the distance as the square root of the sum of the squared differences between the two Numpy arrays of numbers. Which then returns the k most similar neighbours from the training set for a given test instance. Once we have calculated the closest nearest neighbour for a test instance, Ones found be than use the predicted response based on those found neighbours. We we’re able to achieve this by allowing each neighbour to vote for their class attribute and take the majority vote as the prediction.

## Java
In the Java program, I had to first read the CSV file using a scanner, then turn it into an array, turn every value including the strings into integers (e.g. female = 0, male = 1), calculate the Euclidean distance between the test set and the training set, sort the distances in ascending order, print all k nearest neighbours and all predicted classes through the use of HashMaps. The HashMap will show the class and the number of its occurrences in the k nearest neighbours. The total number of occurrences should be equal to k.

Below, is the accuracy of predictions in the KNN algorithm.
* Python K-NN version:
   - K = 5
   - Accuracy: 0.77
* Java K-NN version:
   - K = 1
   - Accuracy = 0.76


# Dataset

We used Adult Income Data which was published in 1996, this data set came with test data which had 45222 instances, which consist of 3620 unknown random missing data which was denoted by a “?”. This data set consisted of 15 features:
*	Age 	
*	Sex
*	fnlwgt	
*	Capital Gain 
*	Education 	
*	Hours per week 
*	Education Num 	
*	Capital Loss 
*	Marital Status 	
*	Native Country 
*	Occupation 	
*	Earns 
*	Relationship 	
*	Working Class 
*	Race	

Moreover, the data has only two classes, under the feature name Earns ,which represents whether an individual earns more than £50,000 or less than £50,000 a year; in order to predict whether the income was indeed higher or lower than 50k, we used classification algorithms to predict, using the data point to see if a person earns more than 50k or not, whereby machine learning algorithms tools such as: K Nearest Neighbours, confusion matrix, 5-Fold Cross Validation were implemented. 


## Cleaning data

One of the first important task was to clean and declutter the missing data and any another data point that consisted of no values. For me to clean the data, I firstly removed 3620 instances that contained the symbol “?”, this is because the missing data would have interrupted the prediction rating. 

Deleting data point isn’t good practice because every data is valuable Although the missing data only consisted of 6% of the overall data points, deleting it would have not made a massive difference and thus would not cause any disruption to the algorithm outcome. The data was further analysed; and many of the features that were irrelevant to the prediction of an individual earning less or more than £50K were removed. The features that I felt needed removing were; 

*	Fnlwgt: These features held large unnecessary numerical values that played no contribution to an individual’s prediction. 
*	Education: This feature consists of the numerical values that represent the education level that the individuals received. This data was unnecessary as it wouldn’t add any benefit in the prediction value of finding out if individual earning is below or above 50K. There is already an education Num feature which represents the same data, so this was kept and used in the algorithm. 
*	Capital Loss and Capital Gain: This feature was deleted because most of the data contained zero, which ultimately made it useless data and thus no longer required. 


The datasets used for KNN:
* [adult.test.csv](https://github.com/ish2nv/KNN-Model/blob/master/KNNModel_CW/Knn%20(Java%20version)/KNNModel/src/adult.test.csv)
* [adult.train.5fold.csv](https://github.com/ish2nv/KNN-Model/blob/master/KNNModel_CW/Knn%20(Java%20version)/KNNModel/src/adult.train.5fold.csv)

