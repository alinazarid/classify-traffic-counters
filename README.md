
## \# Machine Learning Project {#-machine-learning-project}

### Ali Nazari (This version works and it is before implementing PCA)

##### Spring 2020


## Introduction

There are more than 12 thousand miles of roads and highways in state of
Idaho. Idaho Transportation Department (ITD) is in charge maintenance
and development of these roads. One of the metrics used evaluating our
roads is the annual traffic volume count on these roads. The most common
method to collect the traffic count is establish a short-term portable
count station. There are more than 50 thousand of these stations in
Idaho. It is not possible for ITD to collect data from all these
stations every year. The manual task of selecting proper stations at
which a field crew collects the traffic count is tedious and time
consuming. This project is intended to use Machine Learning algorithms
to classify the stations into picked and not picked. This notebook can help
analysts at ITD to make a quicker and more informed decision in picking a
traffic station for traffic count collection.

## Dependencies
Python 3.6
	matplotlib
	pandas
	numpy
	seaborn
	sklearn
	imblearn
	mpl_toolkits

## Execution

1- Clone the repository

2- Create a virtual environment with the above libraries

3- Run the notebook

## Data

The data is gathered from ITD's public data warehouse. After aggregating
the station data and filtering out valid stations, data is exported as
text files. The test files are used as the input files for this
notebook. There are several fairly independent features assumed for this
data set. The categorical features are the following
[Source](https://www.fhwa.dot.gov/policyinformation/hpms/fieldmanual/page00.cfm):

-   RouteID: The method of stations spatial referencing is Linear
    Referencing System, in which the locations of stations along a
    linear element are described in terms of measurements from a fixed
    point. Each linear element is recognized as a route. The routes are
    categorized as: OH (Off Highways), SH (State Highways), IN
    (Interstate Highways) and US (US Highways).

-   FuntionalClass: Functional systems result from the grouping of
    highways by the character of service they provide.

    1.  *Interstate*

    2.  *Principal Arterial -- Other Freeways and Expressways*

    3.  *Principal Arterial -- Other*

    4.  *Minor Arterial*

    5.  *Major Collector*

    6.  *Minor Collector*

    7.  *Local*

-   FacilityType:

    1.  *One-Way Roadway: Roadway that operates with traffic moving in a
        single direction during non-peak period hours.*

    2.  *Two-Way Roadway: Roadway that operates with traffic moving in
        both directions during non-peak period hours.*

    3.  *Ramp: Non-mainline junction or connector facility contained
        within a grade-separated interchange. *

    4.  *Non Mainline: All non-mainline facilities excluding ramps. *

    5.  *Non-Inventory Direction: Individual road/roads of a multi-road
        facility that is/are not used for determining the primary length
        for the facility. *

    6.  *Planned/Unbuilt: Planned roadway that has yet to be
        constructed. *

-   Urban: According to definitions in 23 U.S.C. 101(a), areas of
    population greater than 5,000 qualify as urban for transportation
    purposes.

-   AADT: Annual Average Daily Traffic (AADT) identifies the average
    volume of traffic for the average one day (24-hour period) during a
    data reporting year at a specific location or specific segment of
    road.

-   CAADT: Commercial vehicles AADT.

-   RecentYear: Number of years passed since the last count in that
    station.

-   X_cord and Y_cord: X and Y coordinates of a station.

The above data are the result of merging the following datasets.

-   The stations selected for collecting data during 2019
-   The stations along with the year when the most recent count
    was collected
-   All the valid station for this study excluding the automatic traffic recorders



## Preprocessing

### Cleaning Data

In this step, the correct data type is asigned to each feature and the
features of interest are filtered out of the data.

### Transform Categorical Features

The dataset contains two different data types. Numerical and
Categorical. Typically, any standard workflow in feature engineering
involves some form of transformation of these categorical values into
numeric labels and then applying some encoding scheme on these values.
There are two major classes of categorical data, nominal and ordinal. In
this project I am dealing with purely nominal features
([source](https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63)).

In any nominal categorical data attribute, there is no concept of
ordering amongst the values of that attribute. In our example the
following features are nominal: \* RouteID \* Facility Type \*
Functional class \* Urban

Since the data is consists of both numerical and nominal data, the
nominal categorical data are transferred to one-hot-encoding features.

### Scaling Data

Often machine learning algorithms perform better or converge faster when
features are on a relatively similar scale and/or close to normally
distributed. It is observed in the above plot that the variables
distributions vary and are highly skewed.

**Log transformation** I use this method for transforming AADT and CAADT
feature to force them closed to normal distribution. It seems that using
a log transformation decreased the scale of the distributions. It seems
the outliers caused the log-transformed distributions to still be a bit
skewed, but it is closer to normal than the original distribution.

**MinMaxScaler** Min-Max scaler doesn't reduce the skewness of a
distribution. It simply shifts the distribution to a smaller scale
\[0--1\]. For this reason, it seems Min-Max scaler isn't the best choice
for a distribution with outliers or severe skewness. I am using these
methods to scale the Hypergeometric features which are X_cord and
Y_cord. I am using this scaler for scaling RcentYear feature as well
since it is highly skewed based on the distributions above.

[Source](https://medium.com/@sjacks/feature-transformation-21282d1a3215)


### Train-Test Split

Since this data is extremely imbalanced, in splitting the data the
stratify method is used to make sure the distribution of classed won\'t
be disturbed. The test data is reserved for the last step to check the
accuracy of the data.

### Balancing Data

The data is strangely imbalance and I need to use a sampling technique
to balance the data. The data is imbalance because there are numerous
stations in Idaho that are located in rural and remote areas. The
abundant of these stations in data are caused the imbalance properties
of the data. These stations are seldom selected for data collection
because of the scarcity of traffic and difficulty in accessing their
location; hence they are classified as *NotPicked*. Several methods to
tackle this problem are considered.

-   Under-sample: In this method we are losing data by removing some of
    data points from the dataset. I use the under-sampling method in
    order to exclude some of the majority class (NotPicked) in the
    training data so the results won\'t be affected by them.
    Undersampling is efficient and feasible approach for our case since
    the size of the dataset is large.
-   Over-sample: This method will not help to improve our model in this
    case because it creates only duplications and is not a adding any
    new information for the model so it can differentiate between
    classes.
-   Synthetic Minority Oversampling Technique (SMOTE) This method works
    when the samples are similar to each other, hence considering their
    mean as a new data point is reasonable. But if they are not distinct
    and are spread apart, this method will result in introducing noise
    to the data by creating data points that might be closer to the
    majority class. The PCA plot below proves that this is the case for
    our project.


PCA Visualization suggests that the highest two components are not enough
to explain the variance and there are no few dominant features than can
be used to linearly separate the data. There is no dominant feature
among the independent variables that would explain more than 50% of the
variance. The features might not be quite independent. Also, This
suggests that the dependency between independent variables (predictors)
and the dependent variable (prediction) is not strong.



## Model Selection

I am looking at four different classification tools and their
performance. \* Logistic Regression \* ComplementNB \*
DecisionTreeClassifier \* SVC

Logistic Regression is chosen as the basic linear model to observe its
performance against the non-linear separable data. Logistic Regression
Model is binary classification model that often is used as a baseline
due to its simple performance.

Naive Bayes methods are a set of supervised learning algorithms based on
applying Bayes' theorem with the "naive" assumption of conditional
independence between every pair of features given the value of the class
variable. The different naive Bayes classifiers differ mainly by the
assumptions they make regarding the distribution of P(x_i \\mid y).
ComplementNB implements the complement naive Bayes (CNB) algorithm. CNB
is an adaptation of the standard multinomial naive Bayes (MNB) algorithm
that is particularly suited for imbalanced data sets. Specifically, CNB
uses statistics from the complement of each class to compute the model's
weights. The inventors of CNB show empirically that the parameter
estimates for CNB are more stable than those for MNB.

Decision Trees (DTs) are a non-parametric supervised learning method
used for classification. Requires little data preparation. Other
techniques often require data normalization, dummy variables need to be
created and blank values to be removed. Decision tree learners create
biased trees if some classes dominate. It is therefore recommended to
balance the dataset prior to fitting with the decision tree.

SVC is a strong classification model and when used with rbf kernel can
perform nonlinear classifications with high performance.


## Conclusion

This project attempts to tackle a binary classification task with both
categorical and numerical features. The results show that among four
different classification algorithms (Naive Bayes, Decision Tree,
Logistic Regression and Support Vector Classifier) the SVC out-performed
other methods.

This task main goal was to identify the traffic volume count stations on
Idaho which are likely to get selected for data collection based on the
characteristics of the road where the stations are located and their
location. This task focus is on identifying the True class meaning the
stations that need to get selected rather than stations that are getting
selected. The reason is the fact that through state of Idaho there are
lots of stations in rural areas which are critical. This explains why
the metric to evaluate the models against each other was selected to be
True Positive Rate and the area under R.O.C.

The main challenge is this task is combining these two types of data.
Using one hot encoder I was able to transform the categorical data into
binary features, but the classifiers are often do a better job when they
are working with either discrete binary features or continuous features.
To improve the models is to modify the continuous variables into
categorical variables based on the technical literature. Using all
binary features would allow us to utilize classifiers that performed
well with binary features such as Bernoulli Naive Bayes classifier
instead of Complement Naive Bayes classifier.

A more in-depth data science study needs to be performed to show the
independency of the features. It is possible that some of the
categorical feature might not be quite independent and this causes a
large bias in the classification task.

Since this data is extremely imbalance, another method to improve the
task is to define class weight for out classifiers and make up for the
unbalance data rather than using random under sampling and over
sampling.

Two features in the data had Hypergeometric distributions. I could not
find a proper encoding schema to scale these features hence I used
Min-Max scaling technique. These two features are not able to be
transformed to categorical features because they are representing the
coordinates of statins. These two features need either be removed or
mapped to municipals and then those municipals get transformed into
categorical data.

### Contact
For more information contact (alinazarid@gmail.com)[alinazarid@gmail.com]

