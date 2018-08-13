from sklearn import tree
from sklearn import svm
from sklearn import naive_bayes

"""
Decision Tree (machine learning model):
    flowchart that stores data and asks
    each label data point that it receives a 
    yes or no question

Support Vector Machines (SVMs):
    LinearSVC

Naive Bayes:
    Bayes' theorem with the "naive" assumption of
    independence between every pair of features
    Gaussian

"""

#X = [height, weight, shoe size]
#Y stores a list of labels of gender (stored as strings) that corresponds to each data set in X


X = [[181, 80, 44], [177,70,43], [160,60,38], [154, 54, 37],
     [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70 , 40],
     [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']



"""
clf is a classifier variable that will store our
decision tree model
tree dependency is referenced by class and then method
"""

clf = tree.DecisionTreeClassifier()
clf_1 = svm.LinearSVC()
clf_2 = naive_bayes.GaussianNB()

#fit method trains our decision tree on our variables
clf = clf.fit(X,Y)
clf_1 = clf_1.fit(X,Y)
clf_2 = clf_2.fit(X,Y)
T = [190, 70, 43]

prediction = clf.predict([T])
prediction_1 = clf_1.predict([T])
prediction_2 = clf_2.predict([T])

print("Decision Tree: ", prediction)
print("CLV: ", prediction_1)
print("Gaussian Naive Bayes:", prediction_2)