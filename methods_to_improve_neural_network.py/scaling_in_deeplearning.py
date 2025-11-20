import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score

data = pd.read_csv("record_new.csv")
print(data.head())
print(data.isnull().sum())
# sns.scatterplot(x=data['Age'] ,y= data['EstimatedSalary'])
# plt.show()
X=data.drop(columns=['Purchased'])
y=data['Purchased']

# print(type(X) , type(y))
print(X.head() , y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)
ss= StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)
model= Sequential()
model.add(Dense(128,  activation='relu' ,input_dim=2))
model.add(Dense(1 ,  activation='sigmoid'))
print(model.summary())
callback = EarlyStopping(
  
    monitor='val_loss',
    patience=5,
    verbose=0,
    restore_best_weights=False 
)

# we need feature scaling
model.compile (loss='binary_crossentropy' , optimizer='adam' , metrics=['accuracy'])

history = model.fit(X_train_scaled , y_train , validation_split=0.2 ,epochs= 1000, callbacks = callback)
y_pred_prob = model.predict(X_test_scaled)
y_pred = np.where(y_pred_prob > 0.5, 1, 0)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
plt.plot(history.history['val_accuracy'])
plt.show()

