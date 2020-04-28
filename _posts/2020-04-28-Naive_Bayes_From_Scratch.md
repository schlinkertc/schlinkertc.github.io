---
layout: post
title: "Naive Bayes From Scratch"
date: 2020-04-28
---

# Naive Bayes From Scratch

In an effort to reinforce my understanding of statistical decision making, I wanted to build a naive bayes classifier without using a stats library like Sklearn. In the code that follows, you can see the results of my efforts. I was able to build a classifier that performs as well as sklearn's GaussianNB classifier.  

## Importing and Cleaning the Data

I'm using data from the MLB Stats API that I've previously collected for a [separate, ongoing project](https://github.com/schlinkertc/MLB_Analysis). For this example, we'll use the end speed and spin rate of mlb pitches to classify them as either a fastball or not a fastball. End speed refers to the velocity of the ball as it crosses the plate, and [spin rate](http://m.mlb.com/glossary/statcast/spin-rate) refers to the revolutions per minute of the ball after it's released.

If you want to use this data for your own purposes, you can find it on [GitHub](https://github.com/schlinkertc/Pitch-Classification).


```python
import os
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np
sns.set_style('whitegrid')
%config InlineBackend.figure_format = 'retina'

sns.set(color_codes=True)
sns.set(rc={'figure.figsize':(12,10)})
```


```python
path = os.getcwd().strip('Bayes')+'FetchData/datasets/'

df = pd.read_csv(path+'pitchSpeed_spinRate.csv')
print(df.shape)
df.dropna(inplace=True)

df.head()
```

    (1910986, 3)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>endSpeed</th>
      <th>spinRate</th>
      <th>pitchType</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>87.7</td>
      <td>1926.0</td>
      <td>Four-Seam Fastball</td>
    </tr>
    <tr>
      <th>1</th>
      <td>89.0</td>
      <td>1872.0</td>
      <td>Four-Seam Fastball</td>
    </tr>
    <tr>
      <th>2</th>
      <td>83.9</td>
      <td>1661.0</td>
      <td>Changeup</td>
    </tr>
    <tr>
      <th>3</th>
      <td>88.3</td>
      <td>1934.0</td>
      <td>Two-Seam Fastball</td>
    </tr>
    <tr>
      <th>4</th>
      <td>88.9</td>
      <td>1997.0</td>
      <td>Four-Seam Fastball</td>
    </tr>
  </tbody>
</table>
</div>



Certain pitches aren't useful for our purposes. 
- 'Automatic Balls' are not actually thrown. They're used when a pitcher strategically decides to [intentionally walk](https://en.wikipedia.org/wiki/Intentional_base_on_balls) a batter
- A ['Pitchout'](https://en.wikipedia.org/wiki/Pitchout) is a throw away pitch used when you want to catch a runner stealing 
- An Eephus is a [rare and hilarious](https://www.youtube.com/watch?v=VfWXADedncM) trick pitch


```python
# drop outlier pitch types
pitchTypes_toDrop = ['Automatic Ball','Pitchout','Eephus']
i = df[df['pitchType'].isin(pitchTypes_toDrop)].index
df.drop(index=i,inplace=True)

df.shape
```

To make things simpler, we're only going to classify 'fastball' vs 'not fastball'.


```python
def is_fastball(x):
    if 'Fastball' in x['pitchType']:
        return 'fastball'
    else:
        return 'not fastball'
df['is_fastball'] = df.apply(lambda x: is_fastball(x),axis=1)
df['is_fastball'].value_counts()
```




    not fastball    974122
    fastball        873880
    Name: is_fastball, dtype: int64



### Explore the Data

First, we want to see if our features our normally distributed. That way, we can use the Gaussian Probability Density function to estimate the likelihood of observing a particular value in our feature vectors. 


```python
f, axes = plt.subplots(1,2)
sns.distplot(df['endSpeed'],ax=axes[0])
sns.distplot(df['spinRate'],ax=axes[1])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a239e8e48>




![png](output_12_1.png)


Next, we can use a simple scatter plot to see if our features provide reasonable separation between their respective classes.


```python
fig, ax = plt.subplots()
ax.set(xscale='symlog',yscale='symlog')
sns.scatterplot(x=df['endSpeed'],y=df['spinRate'],hue=df['is_fastball'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x10c11a4a8>




![png](output_14_1.png)


## Building our Classifier

![bayes_image.png](https://miro.medium.com/max/768/1*zPseemLGYHMS8M0phAhhoA.png)

'A' is our class value (fastball vs not fastball) and 'B' are our predictive features (end speed and spin rate)

### Breaking Down Bayes' Theorem

We need to calculate the probability for each class, fastball and not fastball, given our input data. Then, we choose the class that has the highest probability. We can ignore the denominator in this case. We only care about which probability is higher, and the denominator would be same value in both equations.

P(A) is our "prior" probability, and it's very easy to calculate. In our case, it's the number of fastballs (or non-fastballs) divided by the total pitches in the dataset. 


```python
def is_fastball(x):
    if 'Fastball' in x['pitchType']:
        return 'fastball'
    else:
        return 'not fastball'
df['is_fastball'] = df.apply(lambda x: is_fastball(x),axis=1)
df['is_fastball'].value_counts()
```




    not fastball    974122
    fastball        873880
    Name: is_fastball, dtype: int64



P(B|A) is our "liklihood", and that's a little more complicated. We need to determine the liklihood of observing oura particular value in our input vectors given a class value. Since our features our roughly normally distributed, we can use the Gaussian Probability Density Function. We only need the mean and standard deviation of a class vector the calculate liklihood.


```python
# Create a function that calculates p(x | y):
def likelihood(x,mean,sigma):
    return np.exp(-(x-mean)**2/(2*sigma**2))*(1/(np.sqrt(2*np.pi)*sigma))
```

### Building our Classifier

This is a factory function that takes in our input vectors (X) and outputs (Y) and calculates the figures we need to classify a pitch. As stated above, we'll need prior probabilites as well as the mean and the standard deviation for our input feature vectors seperated by class value. The function returns another function. The returning function will only take in the input vectors, and it will return probabilities for each class as a dictionary of NumPy arrays.


```python
def fit(X,Y):
    classes = np.unique(Y)
    num_predictors = X.shape[1]
    
    # params dictionary will store the values we need to predict
    params = {}
    for G in classes:
        class_params = []
        for i in range(num_predictors):
            feature_params = {}
            feature_params['prior'] = len(np.where(Y==G)[0])/len(Y)
            feature_params["mean"] = X[np.where(Y==G)][:,i].mean()
            feature_params["sigma"] = X[np.where(Y==G)][:,i].std()
            
            class_params.append(feature_params)
        params[G]=class_params
    
    # The function that is returned from the factory function retains scope. 
    # So it knows the values calculated above
    def predict(X_test):
        class_probabilites = {}
        for G in classes:
            liklihoods = np.empty(X_test.shape)
            for i in range(liklihoods.shape[1]):
                liklihoods[:,i] = likelihood(X_test[:,i],params[G][i]['mean'],params[G][i]['sigma'])
            out = np.prod(liklihoods,axis=1)
            out*=params[G][0]['prior']
            class_probabilites[G]=out
        return class_probabilites
                
    return predict
```


```python
y = df['is_fastball'].to_numpy()
x = df[['endSpeed','spinRate']].to_numpy()
nb = fit(x,y)
nb
```




    <function __main__.fit.<locals>.predict(X_test)>




```python
predict = nb(x)
predict
```




    {'fastball': array([2.27745304e-05, 7.18554093e-06, 6.92096463e-07, ...,
            9.13635997e-05, 9.01459308e-05, 1.26118320e-04]),
     'not fastball': array([5.06947230e-06, 2.62086247e-06, 8.21903785e-06, ...,
            1.17705429e-05, 1.24653868e-05, 1.38775678e-05])}




```python
predicted_fastballs = predict['fastball']>predict['not fastball']
predicted_fastballs
```




    array([ True,  True, False, ...,  True,  True,  True])



## Evaluating the Results

First, we'll use sklearn's classification report to get a high level summary of how our classifier did when it was trained on the whole dataset.


```python
from sklearn.metrics import classification_report
pd.DataFrame(classification_report(y == "fastball",predicted_fastballs,output_dict=True))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>False</th>
      <th>True</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.921015</td>
      <td>0.793438</td>
      <td>0.850679</td>
      <td>8.572267e-01</td>
      <td>8.606868e-01</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.783955</td>
      <td>0.925057</td>
      <td>0.850679</td>
      <td>8.545062e-01</td>
      <td>8.506793e-01</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.846976</td>
      <td>0.854207</td>
      <td>0.850679</td>
      <td>8.505918e-01</td>
      <td>8.503957e-01</td>
    </tr>
    <tr>
      <th>support</th>
      <td>974122.000000</td>
      <td>873880.000000</td>
      <td>0.850679</td>
      <td>1.848002e+06</td>
      <td>1.848002e+06</td>
    </tr>
  </tbody>
</table>
</div>



It looks pretty good! But just to make sure, let's split the data into train and test sets and compare the results versus sklearn.


```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
#split the data
X_train,X_test,y_train,y_test = train_test_split(x,y, test_size=.3, random_state=0)

#fit
gnb = GaussianNB()
sk_fit = gnb.fit(X_train,y_train)
#predict
sk_y_pred = sk_fit.predict(X_test)

pd.DataFrame(classification_report(y_test,sk_y_pred,output_dict=True))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fastball</th>
      <th>not fastball</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.793633</td>
      <td>0.920872</td>
      <td>0.850783</td>
      <td>0.857252</td>
      <td>0.860725</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.924814</td>
      <td>0.784416</td>
      <td>0.850783</td>
      <td>0.854615</td>
      <td>0.850783</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.854217</td>
      <td>0.847184</td>
      <td>0.850783</td>
      <td>0.850700</td>
      <td>0.850508</td>
    </tr>
    <tr>
      <th>support</th>
      <td>262070.000000</td>
      <td>292331.000000</td>
      <td>0.850783</td>
      <td>554401.000000</td>
      <td>554401.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#fit
my_fit = fit(X_train,y_train)

#predict
my_y_pred = my_fit(X_test)
my_y_pred = my_y_pred['fastball']>my_y_pred['not fastball']

pd.DataFrame(classification_report(y_test=='fastball',my_y_pred,output_dict=True))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>False</th>
      <th>True</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.920872</td>
      <td>0.793633</td>
      <td>0.850783</td>
      <td>0.857252</td>
      <td>0.860725</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.784416</td>
      <td>0.924814</td>
      <td>0.850783</td>
      <td>0.854615</td>
      <td>0.850783</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.847184</td>
      <td>0.854217</td>
      <td>0.850783</td>
      <td>0.850700</td>
      <td>0.850508</td>
    </tr>
    <tr>
      <th>support</th>
      <td>292331.000000</td>
      <td>262070.000000</td>
      <td>0.850783</td>
      <td>554401.000000</td>
      <td>554401.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Conclusion

Our homemade Bayes' classifier returned the same result as sklearn, so I'm happy with this. It works reasonably well in this situation because our input features are roughly normally distributed, and there's good separation between class values. Naive Bayes is also good choice in this particular situation because of how many records we have. Naive Bayes doesn't take very long to produce results because we only need a few data points (mean, standard deviation, and prior probabilities) upon which to form our predictions.  
