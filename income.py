import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

df=pd.read_csv("income.csv")
print(df)

plt.scatter(df["Age"],df["Income"])
plt.show()

km=KMeans(n_clusters=3)
#print(km)

y_pred=km.fit_predict(df[["Age","Income"]])
print(y_pred)

df["cluster"]=y_pred
print(df)


df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

plt.scatter(df1.Age,df1["Income"],color="green")
plt.scatter(df2.Age,df2["Income"],color="red")
plt.scatter(df3.Age,df3["Income"],color="yellow")

plt.xlabel("Age")
plt.ylabel("Income")
plt.legend()
plt.show()

scaler=MinMaxScaler()
scaler.fit(df[["Income"]])
df["Income"]=scaler.transform(df[["Income"]])
print(df)

scaler.fit(df[["Age"]])
df["Age"]=scaler.transform(df[["Age"]])
print(df)


km=KMeans(n_clusters=3)
y_pred=km.fit_predict(df[["Age","Income"]])
print(y_pred)
df["cluster"]=y_pred
print(df)


df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

plt.scatter(df1.Age,df1["Income"],color="green")
plt.scatter(df2.Age,df2["Income"],color="red")
plt.scatter(df3.Age,df3["Income"],color="yellow")

plt.xlabel("Age")
plt.ylabel("Income")
plt.legend()
plt.show()

centroids=km.cluster_centers_
print(centroids)

df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

plt.scatter(df1.Age,df1["Income"],color="green")
plt.scatter(df2.Age,df2["Income"],color="red")
plt.scatter(df3.Age,df3["Income"],color="yellow")
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label="centroid")

plt.xlabel("Age")
plt.ylabel("Income")
plt.legend()
plt.show()

#Elbow Plot

k_rng=range(1,10)
sse=[]
for k in k_rng:
	km=KMeans(n_clusters=k)
	km.fit(df[["Age","Income"]])
	sse.append(km.inertia_)
print(sse)

plt.xlabel("k")
plt.ylabel("Sum of Squared Error")
plt.plot(k_rng,sse)
plt.show()


