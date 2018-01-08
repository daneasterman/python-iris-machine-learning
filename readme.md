## Introduction to Machine Learning and Data Science Analysis Techniques with Python

In this article, we will be looking at the classic Iris flower dataset, which is widely considered to be the fundamental "hello world" example to demonstrate data science and machine learning concepts with the Python programming language. 

This readme provides an explanation behind the main `iris-final.py` file in this repository which incorporates the main machine learning code snippets outlined below.

However, I would recommend working through all the steps to gain a better understanding of the subject.

### The "Pandas" Python Library

*__What Does Pandas Do?__*

Pandas offers a number of easy-to-use methods or operations to manually view, analyse and understand data in the mould of a traditional data scientist. This is an important step before applying any automated machine learning techniques.

*__Key Examples__*

If working in the Python "interpreter" (simply type `python` to access this) within your terminal, the first thing you need to do is import the Pandas libraries and declare basic variables such as `url`, `names` and `dataset`:

```
import pandas
from pandas.tools.plotting import scatter_matrix

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
```

Now by running Python's print command, and applying the `head` method to the `dataset` variable, we can see a tabulated summary of the first 20 entries from our Iris flower data. This is useful in order to get a quick snapshot of the data.

```
print(dataset.head(20))
```

The result should be to "print" out the following in your terminal:

```
    sepal-length  sepal-width  petal-length  petal-width        class
0            5.1          3.5           1.4          0.2  Iris-setosa
1            4.9          3.0           1.4          0.2  Iris-setosa
2            4.7          3.2           1.3          0.2  Iris-setosa
3            4.6          3.1           1.5          0.2  Iris-setosa
4            5.0          3.6           1.4          0.2  Iris-setosa
5            5.4          3.9           1.7          0.4  Iris-setosa
6            4.6          3.4           1.4          0.3  Iris-setosa
7            5.0          3.4           1.5          0.2  Iris-setosa
8            4.4          2.9           1.4          0.2  Iris-setosa
9            4.9          3.1           1.5          0.1  Iris-setosa
10           5.4          3.7           1.5          0.2  Iris-setosa
11           4.8          3.4           1.6          0.2  Iris-setosa
12           4.8          3.0           1.4          0.1  Iris-setosa
13           4.3          3.0           1.1          0.1  Iris-setosa
14           5.8          4.0           1.2          0.2  Iris-setosa
15           5.7          4.4           1.5          0.4  Iris-setosa
16           5.4          3.9           1.3          0.4  Iris-setosa
17           5.1          3.5           1.4          0.3  Iris-setosa
18           5.7          3.8           1.7          0.3  Iris-setosa
19           5.1          3.8           1.5          0.3  Iris-setosa
```

Another simple but very useful method from the pandas library is `describe`. This automatically provides us with some basic statistical information such as the mean, standard deviation (std), minimum/maximum ranges and some percentiles:

```
print(dataset.describe())
```

This results in:

```
       sepal-length  sepal-width  petal-length  petal-width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.054000      3.758667     1.198667
std        0.828066     0.433594      1.764420     0.763161
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
```

###  Matplotlib and Data Visualisation

The Matplotlib official website describes the software as a "Python 2D plotting library". In other words it allows us to create simple data "visualisations" or graphs/charts of our data.

In this example Pandas is used in conjunction with Matplotlib to generate four histogram graphs:

First ensure the Matplotlib library is imported:

```
import matplotlib.pyplot as plt
```

Then run the `hist` method on our dataset variable followed by `show()` on `plt` which is the shorthand representation of our use of the Matplotlib library here:

```
dataset.hist()
plt.show()
```

After running the command above, the histogram image should automatically open and display as a png image file:

![Histogram](https://i.imgur.com/Mb9x5O7.png)

### Machine Learning with Scikit-Learn

This is where the code starts to get a bit more complex but also more interesting!

Scikit-Learn is a very well-established open source machine learning library (10 years old), and it's still in active developement! It has many sophisticated uses in the fields of data analyis and data mining, but in this example we will be using it to automate the process of flower classification.

In other words, we will train the algorithm to "learn" or categorise the correct flower species based on the the petal and sepal measurements of a particular data entry.

*__Assessing which Machine learning algorithm to use__*

Due to the added complexity, at this point I would recommend moving away from working in Python interepreter and putting your code in a simple code editor such as Sublime and then running the individual file instead of each line of code.

The first major step we are going to take is compare the various Machine Learning algorithms at our disposal to see which one is best for the job.

Before that we need to add all the necessary import statements for Scikit-Learn (sklearn) below our imports for pandas and matplotlib. So the full import section of the code will look like this:

```
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
```

Next we need to separate out the dataset according to a 80/20% split. 80% will be used to train the algorithm as "seen" data while the remaining 20% will be used as "unseen" data to test or validate how accurately the algorithm has classified the flowers.

```
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
dataset = pandas.read_csv(url, names=names)

array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

seed = 7
scoring = 'accuracy'
```

Now we "append" 6 different algorithms to an empty models array, and use a for loop to "loop/iterate over" and compare the accuracy of each one in turn:

```
# Check accuracy of algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
  kfold = model_selection.KFold(n_splits=10, random_state=seed)
  cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)
```

Add all three of the above code snippets together, and if everthing is working correctly we will see the following printout showing the estimated accuracy score of each algorithm:

```
LR: 0.966667 (0.040825)
LDA: 0.975000 (0.038188)
KNN: 0.983333 (0.033333)
CART: 0.966667 (0.040825)
NB: 0.975000 (0.053359)
SVM: 0.991667 (0.025000)
```

Here we can see that with a score of `0.983333`, the K-Nearest Neighbors (KNN) algorithm has performed the best and most accurately on our Iris flower dataset.

*__Making an actual prediction with Machine Learning:__*

Now that we know that KNN is the most accurate algorithm, we can run it exclusively on the data with the code below:

```
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
```

This generates a final average accuracy score of 0.9 (90%) and a classification report breaking down the level of precision or accuracy for each flower species:

```                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00         7
Iris-versicolor       0.85      0.92      0.88        12
 Iris-virginica       0.90      0.82      0.86        11

    avg / total       0.90      0.90      0.90        30
```

### Credits and references

The following resources were invaluable to me in preparing this article on Python and Machine Learning:

[https://medium.com/botsupply/machine-learning-the-conjuring-code-episode-1-c8145d9d67e4]()

[http://blog.fastforwardlabs.com/2016/02/24/hello-world-in-keras-or-scikit-learn-versus.html]()

[https://machinelearningmastery.com/machine-learning-in-python-step-by-step/]()

[https://www.scipy.org/install.html]()

[http://www.codeastar.com/beginner-data-science-tutorial/]()




