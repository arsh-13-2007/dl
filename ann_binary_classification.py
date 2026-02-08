import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


df = pd.read_csv("Churn_Modelling.csv")
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

y = df['Exited']
X = df.drop(columns=['Exited'])

ohe = OneHotEncoder(drop='first')
X_cat = ohe.fit_transform(X[['Geography', 'Gender']]).toarray()


X_num = X.drop(columns=['Geography', 'Gender']).values


X_final = np.hstack((X_cat, X_num))

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = Sequential()
model.add(Dense(5, activation="relu",input_dim=X_train_scaled.shape[1]))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train_scaled, y_train, epochs=20)


y_pred_prob = model.predict(X_test_scaled)
y_pred = np.where(y_pred_prob > 0.5, 1, 0)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print(history.history)

plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.show()
