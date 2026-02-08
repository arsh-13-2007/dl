import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf 
import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout 
from sklearn.metrics import accuracy_score , confusion_matrix


df = pd.read_csv("Churn_Modelling.csv")
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())
print(df.head())
print(df['Gender'].value_counts())

df['Gender'] = df['Gender'].map({'Female' :0 , 'Male' : 1 } )
df['Geography'] = df['Geography'].map({'France' :0 , 'Germany' : 1 , 'Spain' : 2  } )
# print(df.head())

print(df['Geography'].unique())
print(df['Geography'].count())
print(df.head())


# sns.boxplot(df['CreditScore'])
# plt.show()

print(df.shape)

X = df.iloc[:, 3:13]  
y = df['Exited']    

print(X.head())
print(y.head())

X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.2 , random_state=0)


ss = StandardScaler()
X_train= ss.fit_transform(X_train)
X_test=ss.transform(X_test)


model = Sequential()

model.add(Dense(6, activation='relu', input_dim=X_train.shape[1]))
model.add( Dense(6 , activation='relu' ))
model.add( Dense(1 , activation='sigmoid' ))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=20)


y_pred_prob = model.predict(X_test)
y_pred = np.where(y_pred_prob > 0.5, 1, 0)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print(history.history)

plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.show()
