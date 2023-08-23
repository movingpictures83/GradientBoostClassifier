import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV


# In[328]:

class GradientBoostRegressorPlugin:
 def input(self, inputfile):
  ################################################ Preprocessing #########################################################
  self.data_path = inputfile

 def run(self):
     pass

 def output(self, outputfile):
  #categorical_cols = ["Race_Ethnicity"]


  data_df = pd.read_csv(self.data_path)

  # # Tramsform categorical data to categorical format:
  # for category in categorical_cols:
  #     data_df[category] = data_df[category].astype('category')
  #

  # Clean numbers:
  #"Cocain_Use": {"yes":1, "no":0},
  cleanup_nums = { "Cocain_Use": {"yes":1, "no":0},
                 "race": {"White":1, "Black":0, "BlackIsraelite":0, "Latina":1},
  }

  data_df.replace(cleanup_nums, inplace=True)

  # Drop id column:
  data_df = data_df.drop(["pilotpid"], axis=1)

  # remove NaN:
  data_df = data_df.fillna(0)

  # Standartize variables
  from sklearn import preprocessing
  names = data_df.columns
  scaler = preprocessing.StandardScaler()
  data_df_scaled = scaler.fit_transform(data_df)
  data_df_scaled = pd.DataFrame(data_df_scaled, columns=names)


  # In[303]:


  ################################################ Users vs Non-Users #########################################################

  # Random Forest
  # Benchmark

  y_col = "Cocain_Use"
  test_size = 0.3
  validate = True

  y = data_df[y_col]

  X = data_df_scaled.drop([y_col], axis=1)

  # Create random variable for benchmarking
  X["random"] = np.random.random(size= len(X))

  X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = test_size, random_state = 2)



  ridge_params = {"learning_rate":[1, 0.5, 0.25, 0.1, 0.05, 0.01],
  "n_estimators":[1, 2, 4, 8, 16, 32, 64, 100, 200],
  "max_depth":np.linspace(1, 32, 32, endpoint=True)
  }
  clf = GradientBoostingRegressor()
  ln_random = RandomizedSearchCV(estimator = clf, param_distributions = ridge_params, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
  # Fit the random search model
  ln_random.fit(X_train, y_train)

  ln_best = ln_random.best_estimator_
  ln_best.fit(X_train, y_train)
  print(ln_best)
  print('Training R^2: {:.2f} \nTest R^2: {:.2f}'.format(ln_best.score(X_train, y_train), ln_best.score(X_valid, y_valid)))

  scores = cross_val_score(ln_best, X, y, cv=5)
  print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

