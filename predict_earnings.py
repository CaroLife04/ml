import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
X_input = np.array

sc = StandardScaler()


dataset = pd.read_csv("pred.csv")

X = dataset.iloc[:, 2:5].values
y = dataset.iloc[:, -1].values
y = y.reshape(-1,1)
# encoder = LabelEncoder()

# X[:,0] = encoder.fit_transform(X[:,0])
# X[:,1] = encoder.fit_transform(X[:,1])
X_encoded = X[:,0:2]
encoded_X = OneHotEncoder(handle_unknown='ignore',sparse = False).fit(X_encoded)
transformedX = encoded_X.transform(X_encoded)
X = np.concatenate((transformedX, X[:,2].reshape(-1,1)),axis=1)
s_c = StandardScaler()
# X_loc = s_c.fit_transform(X[:,2].reshape(-1,1))
y = s_c.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20 , random_state = 0)

regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
r2_score(y_test, y_pred)
