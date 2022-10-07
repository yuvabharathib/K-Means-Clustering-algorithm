# Implementation of K-Means Clustering Algorithm
## Aim
To write a python program to implement K-Means Clustering Algorithm.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation

## Algorithm:

### Step1
Import the necessary packages.

### Step2
Read the csv file.

### Step3
Scatter plot the applicant income and loan amount.

### Step4
Obtain the Kmean clustring for 2 classes.

### Step5
Predict the cluster group of Applicant Income and Loanamount.

## Program:
```python
## Developed by : yuvabharathib
## REGISTER NUMBER : 22002787

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv('clustering.csv')
print(data.head(2))
x1=data.loc[:,['ApplicantIncome','LoanAmount']]
print(x1.head(2))
X=x1.values
sns.scatterplot(X[:,0],X[:,1])
plt.xlabel('Income')
plt.ylabel('Loan')
plt.show()
Kmean=KMeans(n_clusters=4)
Kmean.fit(X)
print('Cluster centers:',Kmean.cluster_centers_)
print('Label:',Kmean.labels_)
predicted_cluster=Kmean.predict([[9200,110]])
print('The cluster group for the ApplicantIncome 9200 and Loan Amount 110 is ',predicted_cluster)
```

## Output:

![Screenshot from 2022-10-01 18-58-42](https://user-images.githubusercontent.com/113497333/193411889-491bc49c-51e2-45f9-beaa-dd120328ca2f.png)

## Result
Thus the K-means clustering algorithm is implemented and predicted the cluster class using python program
