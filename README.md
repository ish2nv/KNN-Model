# Dataset

We used Adult Income Data which was published in 1996, this data set came with test data which had 45222 instances, which consist of 3620 unknown random missing data which was denoted by a “?”. This data set consisted of 15 features:
*	Age 	*	Sex
*	fnlwgt	*	Capital Gain 
*	Education 	*	Hours per week 
*	Education Num 	*	Capital Loss 
*	Marital Status 	*	Native Country 
*	Occupation 	*	Earns 
*	Relationship 	*	Working Class 
*	Race	

Moreover, the data has only two classes, under the feature name Earns ,which represents whether an individual earns more than £50,000 or less than £50,000 a year; in order to predict whether the income was indeed higher or lower than 50k, we used classification algorithms to predict, using the data point to see if a person earns more than 50k or not, whereby machine learning algorithms tools such as: K Nearest Neighbours, confusion matrix, 5-Fold Cross Validation were implemented. 
Cleaning data
One of the first important task was to clean and declutter the missing data and any another data point that consisted of no values. For me to clean the data, I firstly removed 3620 instances that contained the symbol “?”, this is because the missing data would have interrupted the prediction rating. 
Deleting data point isn’t good practice because every data is valuable Although the missing data only consisted of 6% of the overall data points, deleting it would have not made a massive difference and thus would not cause any disruption to the algorithm outcome. The data was further analysed; and many of the features that were irrelevant to the prediction of an individual earning less or more than £50K were removed. The features that I felt needed removing were; 
•	Fnlwgt: These features held large unnecessary numerical values that played no contribution to an individual’s prediction. 
•	Education: This feature consists of the numerical values that represent the education level that the individuals received. This data was unnecessary as it wouldn’t add any benefit in the prediction value of finding out if individual earning is below or above 50K. There is already an education Num feature which represents the same data, so this was kept and used in the algorithm. 
•	Capital Loss and Capital Gain: This feature was deleted because most of the data contained zero, which ultimately made it useless data and thus no longer required. 






The datasets used for KNN:
* [adult.test.csv](https://github.com/ish2nv/KNN-Model/blob/master/KNNModel_CW/Knn%20(Java%20version)/KNNModel/src/adult.test.csv)
* [adult.train.5fold.csv](https://github.com/ish2nv/KNN-Model/blob/master/KNNModel_CW/Knn%20(Java%20version)/KNNModel/src/adult.train.5fold.csv)

