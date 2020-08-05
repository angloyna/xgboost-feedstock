import pandas as pd 
import numpy as np 
import xgboost as xgb 


df = pd.DataFrame({"X1": np.random.rand(100), "X2": np.random.rand(100), "Y": np.random.binomial(n=1, p=0.5, size=100)})

model = xgb.XGBClassifier(tree_method = 'gpu_hist', gpu_id=2, n_gpus=2)

model.fit(X=df.iloc[:,:2], y=df['Y'])
print(model)
print('finished')
