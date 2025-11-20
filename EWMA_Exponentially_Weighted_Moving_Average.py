import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 


df = pd.read_csv("DailyDelhiClimateTest.csv")
# print(df.head())

df= df[['date' , 'meantemp']]

print( df.head())
print(df.isnull().sum())
plt.scatter(y='meantemp' ,x='date' , data=df,  c='red')
plt.show()

# ewma 

x1 = df['meantemp'].ewm(alpha=0.9).mean()       # important 
print(x1)

df['ewm'] = x1 

print(df.shape)

plt.scatter(y='meantemp' ,x='date' , data=df,  c='red')
plt.plot(df['ewm'])
plt.show()