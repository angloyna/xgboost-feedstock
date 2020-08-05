"""
A simple test for xgboost based on scikit-learn.

Note that xgboost's internal tests are NOT shipped with their
python package on PyPI. Hence our own test is rolled here.
"""
import xgboost
import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
import pandas as pd 
import numpy as np 
import xgboost as xgb 

X, y = sklearn.datasets.load_iris(return_X_y=True)
Xtrn, Xtst, ytrn, ytst = sklearn.model_selection.train_test_split(
    X, y, train_size=0.8, random_state=4)

clf = xgboost.XGBClassifier(
    max_depth=2,
    learning_rate=1,
    n_estimators=10,
    silent=True,
    objective='multi:softmax',
    tree_method='gpu_hist',
    seed=5)
clf.fit(Xtrn, ytrn)
ypred = clf.predict(Xtst)
acc = sklearn.metrics.accuracy_score(ytst, ypred)

print('xgboost accuracy on iris:', acc)
assert acc > 0.9


df = pd.DataFrame({"X1": np.random.rand(100), "X2": np.random.rand(100), "Y": np.random.binomial(n=1, p=0.5, size=100)})

model = xgboost.XGBClassifier(tree_method = 'gpu_hist', gpu_id=0, n_gpus=1)

model.fit(X=df.iloc[:,:2], y=df['Y'])

print(model)

print('finished')
