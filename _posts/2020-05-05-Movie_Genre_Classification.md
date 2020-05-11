---
layout: post
title: "Movie Genre Classification Using NLP"
date: 2020-05-05 
---

# Using a Movie's Plot Description to Classify Genre


The goal is to use natural language processing to create a model that predicts a movie's genre using it's plot summary. We have a dataset of 10,000 movies, each of which is classified as one of nine genres. To prepare the data for modeling, we'll use sklearn's CountVectorizer TF-IDF transformer. The countvectorizer uses a custom lemmatizer built with NLTK's WordNetLemmatizer. There is a significant class imbalance; certain genres are more prevelant than others. We will address this using SMOTE, or Synthetic Minority Over Sampling Technique, and a stratified Kfold split during the model cross-validation process. Finally, we'll tune the model's parameters using sklearn's RandomSearch and evaluate our results on a previously unseen test set.

You can check out all of the code and download the data [here](https://github.com/schlinkertc/Movie-Plot-Classification).

### Import Libraries and Load the Data


```python
import pandas as pd
#import text_processing as text
from sklearn import metrics
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
```


```python
df = pd.read_csv("movie_train.csv",index_col=0,)

df.reset_index(drop=False,inplace=True)
df.rename(mapper={'index':'ID'},axis=1,inplace=True)

X = df['Plot']
y = df['Genre']

print(df.shape)
df.head()
```

    (10682, 7)





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
      <th>ID</th>
      <th>Release Year</th>
      <th>Title</th>
      <th>Plot</th>
      <th>Director</th>
      <th>Cast</th>
      <th>Genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10281</td>
      <td>1984</td>
      <td>Silent Madness</td>
      <td>A computer error leads to the accidental relea...</td>
      <td>Simon Nuchtern</td>
      <td>Belinda Montgomery, Viveca Lindfors</td>
      <td>horror</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7341</td>
      <td>1960</td>
      <td>Desire in the Dust</td>
      <td>Lonnie Wilson (Ken Scott), the son of a sharec...</td>
      <td>Robert L. Lippert</td>
      <td>Raymond Burr, Martha Hyer, Joan Bennett</td>
      <td>drama</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10587</td>
      <td>1986</td>
      <td>On the Edge</td>
      <td>A gaunt, bushy-bearded, 44-year-old Wes Holman...</td>
      <td>Rob Nilsson</td>
      <td>Bruce Dern, Pam Grier</td>
      <td>drama</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25495</td>
      <td>1988</td>
      <td>Ram-Avtar</td>
      <td>Ram and Avtar are both childhood best friends....</td>
      <td>Sunil Hingorani</td>
      <td>Sunny Deol, Anil Kapoor, Sridevi</td>
      <td>drama</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16607</td>
      <td>2013</td>
      <td>Machete Kills</td>
      <td>Machete Cortez (Danny Trejo) and Sartana River...</td>
      <td>Robert Rodriguez</td>
      <td>Danny Trejo, Michelle Rodriguez, Sofía Vergara...</td>
      <td>action</td>
    </tr>
  </tbody>
</table>
</div>



### Tokenizing, Lemmatizing and a TF-IDF Transformer

To prepare text documents for Machine Learning pipelines, we need to convert the documents into a matrix of word frequencies. The result is a sparse matrix with columns representing every word in the dataset and rows containing the frequency of each word in a particular document. Sklearn's CountVectorizer can perform this operation in the context of a ML pipeline. 

The CountVectorizer object as a 'tokenizer' argument that can take in a custom tokenizer lemmatizer. Lemmatizing refers to the process of breaking words down to their root. For example, 'running' and 'runs' are counted as the same token for the classification algorithm. To improve the performance of our model, we'll use a lemmatizer built with NLTK, a library that was specifically built for NLP. 

After we've lemmatized and tokenized the documents to create our sparse matrix of word frequencies, we still need to control for documents that are longer than others. If one document is significantly shorter than another, the comparison of word frequencies won't yield compelling results. To fix this, we'll use a TF-IDF transformer that weights the word frequencies according to document length. 

### SMOTE and SGDClassifier

Once the data has been prepared for modeling, we'll want to account for the class imbalance using [synthetic oversampling](https://www.youtube.com/watch?v=FheTDyCwRdE), or SMOTE. 

Finally, we're ready to fit the data and make our predictions. I chose sklearn's Stochastic Gradient Descent Classifier because it takes a shorter time to converge than other models, and it offers several different loss functions that we can compare during the tuning process.

The following code encompasses all of the steps I've just described.


```python
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 

# Custom Lemmatizer
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
    
### Make the SMOTE Pipeline
smote_pipeline = make_pipeline(CountVectorizer(tokenizer=LemmaTokenizer()),
                         TfidfTransformer(),
                         SMOTE(n_jobs=-1,random_state=42),
                         SGDClassifier(n_jobs=-1,verbose=0,random_state=42)
                        )
```

### Tuning the Model

Now that we have our pipeline, let's see how effective its predictions are. The following function returns the cross-validated results of our model by taking in the number of splits for a stratified Kfold cross-validation, plot descriptions as our input vector (X), genres as our targets (Y), and our pipeline. I used a stratified Kfold split to ensure that each fold has the same proportion of classes before the over-sampling step. 


```python
def pipeline_cv(splits, X, Y, pipeline):
    
    kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=777)
    
    reports = []
    for train, test in kfold.split(X, Y):
        fit = pipeline.fit(X.iloc[train], Y.iloc[train])
        prediction = fit.predict(X.iloc[test])
        
        reports.append(
            pd.DataFrame(
                metrics.classification_report(
                    Y.iloc[test],prediction,output_dict=True
                )
            )
        )

    df_concat = pd.concat([x for x in reports])

    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()

    return df_means

```


```python
pipeline_cv(5,X,y,smote_pipeline)
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
      <th>action</th>
      <th>adventure</th>
      <th>comedy</th>
      <th>crime</th>
      <th>drama</th>
      <th>horror</th>
      <th>romance</th>
      <th>thriller</th>
      <th>western</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>f1-score</th>
      <td>0.508273</td>
      <td>0.492725</td>
      <td>0.640416</td>
      <td>0.342903</td>
      <td>0.541284</td>
      <td>0.695955</td>
      <td>0.427871</td>
      <td>0.280562</td>
      <td>0.798318</td>
      <td>0.55907</td>
      <td>0.525367</td>
      <td>0.557589</td>
    </tr>
    <tr>
      <th>precision</th>
      <td>0.449549</td>
      <td>0.422445</td>
      <td>0.624285</td>
      <td>0.290811</td>
      <td>0.711246</td>
      <td>0.607127</td>
      <td>0.355346</td>
      <td>0.290636</td>
      <td>0.704056</td>
      <td>0.55907</td>
      <td>0.495056</td>
      <td>0.589748</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.585542</td>
      <td>0.592266</td>
      <td>0.658962</td>
      <td>0.420886</td>
      <td>0.437666</td>
      <td>0.817857</td>
      <td>0.539296</td>
      <td>0.271533</td>
      <td>0.921905</td>
      <td>0.55907</td>
      <td>0.582879</td>
      <td>0.559070</td>
    </tr>
    <tr>
      <th>support</th>
      <td>166.000000</td>
      <td>66.200000</td>
      <td>544.800000</td>
      <td>65.600000</td>
      <td>754.000000</td>
      <td>168.000000</td>
      <td>129.800000</td>
      <td>137.000000</td>
      <td>105.000000</td>
      <td>0.55907</td>
      <td>2136.400000</td>
      <td>2136.400000</td>
    </tr>
  </tbody>
</table>
</div>



The result of our cross-validated model funtion gives us an overview of classification metrics for each genre. To further tune our model parameters, we'll focus on the weighted F1 score.


```python
### Create scorer
scorer = metrics.make_scorer(metrics.f1_score, average = 'weighted')
```


```python
### Tuning with Random Search

params = {
    'countvectorizer__ngram_range':[(1,2),(1,3)],
    'countvectorizer__max_df':np.linspace(.5,.7,5),
    'countvectorizer__min_df':[1,2,3,4],
    'tfidftransformer__use_idf':[True],
    'tfidftransformer__smooth_idf':[True],
    'sgdclassifier__alpha':np.linspace(.00005,.0002),
    'sgdclassifier__loss':['squared_hinge']
}

random_search = RandomizedSearchCV(smote_pipeline,params,cv=5,n_jobs=-1,scoring=scorer,verbose=0)

pipeline_cv(5,X,y,random_search)
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
      <th>action</th>
      <th>adventure</th>
      <th>comedy</th>
      <th>crime</th>
      <th>drama</th>
      <th>horror</th>
      <th>romance</th>
      <th>thriller</th>
      <th>western</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>f1-score</th>
      <td>0.541071</td>
      <td>0.521904</td>
      <td>0.669948</td>
      <td>0.344754</td>
      <td>0.645931</td>
      <td>0.737875</td>
      <td>0.447168</td>
      <td>0.289791</td>
      <td>0.835373</td>
      <td>0.618421</td>
      <td>0.559313</td>
      <td>0.612447</td>
    </tr>
    <tr>
      <th>precision</th>
      <td>0.520969</td>
      <td>0.540179</td>
      <td>0.659452</td>
      <td>0.431400</td>
      <td>0.648325</td>
      <td>0.692932</td>
      <td>0.432808</td>
      <td>0.376150</td>
      <td>0.793755</td>
      <td>0.618421</td>
      <td>0.566219</td>
      <td>0.611383</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.563855</td>
      <td>0.507689</td>
      <td>0.681355</td>
      <td>0.289790</td>
      <td>0.643767</td>
      <td>0.790476</td>
      <td>0.465259</td>
      <td>0.236496</td>
      <td>0.883810</td>
      <td>0.618421</td>
      <td>0.562500</td>
      <td>0.618421</td>
    </tr>
    <tr>
      <th>support</th>
      <td>166.000000</td>
      <td>66.200000</td>
      <td>544.800000</td>
      <td>65.600000</td>
      <td>754.000000</td>
      <td>168.000000</td>
      <td>129.800000</td>
      <td>137.000000</td>
      <td>105.000000</td>
      <td>0.618421</td>
      <td>2136.400000</td>
      <td>2136.400000</td>
    </tr>
  </tbody>
</table>
</div>




```python
best_model = random_search.best_estimator_
best_model
```




    Pipeline(memory=None,
             steps=[('countvectorizer',
                     CountVectorizer(analyzer='word', binary=False,
                                     decode_error='strict',
                                     dtype=<class 'numpy.int64'>, encoding='utf-8',
                                     input='content', lowercase=True,
                                     max_df=0.6499999999999999, max_features=None,
                                     min_df=3, ngram_range=(1, 3),
                                     preprocessor=None, stop_words=None,
                                     strip_accents=None,
                                     token_pattern='(?u)\\b\\w\\w+\\b',
                                     tok...
                     SGDClassifier(alpha=0.00011428571428571428, average=False,
                                   class_weight=None, early_stopping=False,
                                   epsilon=0.1, eta0=0.0, fit_intercept=True,
                                   l1_ratio=0.15, learning_rate='optimal',
                                   loss='squared_hinge', max_iter=1000,
                                   n_iter_no_change=5, n_jobs=-1, penalty='l2',
                                   power_t=0.5, random_state=42, shuffle=True,
                                   tol=0.001, validation_fraction=0.1, verbose=0,
                                   warm_start=False))],
             verbose=False)



### Evaluating our Model

Now that we have a tuned model, we're ready to dive deeper into some performance metrics. We have a set of observations that we previously set aside to ensure that we aren't over-fitting anything. We previously fit the model using Kfolds, but now we'll fit our model on all of the training data, apply it to our test set for predictions, and evaluate the results.


```python
train_set = pd.read_csv('datasets/movie_train.csv',index_col=0)

X_train = train_set['Plot']
y_train = train_set['Genre']

X_test = pd.read_csv('datasets/movie_test.csv',index_col=0)['Plot']
y_test = pd.read_csv('datasets/test_actuals.csv',index_col=0,header=None,names=['genre'])['genre']

[data.sort_index(inplace=True) for data in [X_test,X_train,y_test,y_train]]

print(X_test.shape,y_test.shape)
```

    (3561,) (3561,)



```python
fit = best_model.fit(X_train,y_train)
y_pred = fit.predict(X_test)
```


```python
report = pd.DataFrame(
    metrics.classification_report(y_test,y_pred,output_dict=True)
)
report
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
      <th>action</th>
      <th>adventure</th>
      <th>comedy</th>
      <th>crime</th>
      <th>drama</th>
      <th>horror</th>
      <th>romance</th>
      <th>thriller</th>
      <th>western</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.477352</td>
      <td>0.515789</td>
      <td>0.697826</td>
      <td>0.423913</td>
      <td>0.656832</td>
      <td>0.750779</td>
      <td>0.457143</td>
      <td>0.433333</td>
      <td>0.803030</td>
      <td>0.638585</td>
      <td>0.579555</td>
      <td>0.632100</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.548000</td>
      <td>0.480392</td>
      <td>0.688103</td>
      <td>0.325000</td>
      <td>0.686688</td>
      <td>0.803333</td>
      <td>0.468293</td>
      <td>0.274262</td>
      <td>0.873626</td>
      <td>0.638585</td>
      <td>0.571966</td>
      <td>0.638585</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.510242</td>
      <td>0.497462</td>
      <td>0.692930</td>
      <td>0.367925</td>
      <td>0.671429</td>
      <td>0.776167</td>
      <td>0.462651</td>
      <td>0.335917</td>
      <td>0.836842</td>
      <td>0.638585</td>
      <td>0.572396</td>
      <td>0.633465</td>
    </tr>
    <tr>
      <th>support</th>
      <td>250.000000</td>
      <td>102.000000</td>
      <td>933.000000</td>
      <td>120.000000</td>
      <td>1232.000000</td>
      <td>300.000000</td>
      <td>205.000000</td>
      <td>237.000000</td>
      <td>182.000000</td>
      <td>0.638585</td>
      <td>3561.000000</td>
      <td>3561.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Confusion Matrix

The confusion matrix can help us see where our model is going wrong by plotting predicted class values vs actual class values. It looks like 'Drama' and 'Comedy' are getting confused for each other, 'Thriller', 'Adventure', and 'Crime' movies are hard to pin down, often being confused for 'Drama'. Ultimately, there are at least as many correct predictions than false predictions for each row of the matrix, so I'm happy with these results. Especially when considering that genre classification is a tricky task for many humans as well. 


```python
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(true,predicted,classes):
    import itertools
    cm=confusion_matrix(true,predicted,labels=classes)
    
    fig = plt.figure(figsize=(15,9))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm,cmap=plt.cm.Blues)
    plt.title('Confusion matrix',fontdict={'size':20})
    fig.colorbar(cax)
    
    ax.set_xticklabels([''] + classes,fontdict={'size':14})
    ax.set_yticklabels([''] + classes,fontdict={'size':14})
    
    plt.xlabel('Predicted',fontdict={'size':14})
    plt.ylabel('True',fontdict={'size':14})
    
    plt.grid(b=None)
    fmt = 'd'

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             fontdict={'size':14,'weight':'heavy'})
```


```python
classes = list(report.columns)[:-3]
plot_confusion_matrix(y_test,y_pred,classes)
```


![png](output_28_0.png)


### Further Considerations

How can we make this model better? We can re-examine our tokenizing and lemmatizing. We only really did the bear minimum on that step. We could continue tuning the parameters of the model with grid-search. 

Most importantly, we could choose a different metric to tune or model after taking into consideration what this model could be useful for. Maybe we want to use this model to recommend movies to users. If we know a user likes comedy, we might want to optimize for recall so that we can be sure that every comedy is represented in the output even if a few non-comedies make it through. If a user only likes drama and hates comdedy, we could optimize for precision to be sure that no comedies make it through. I chose weighted-F1 because it's a balanced metric between precision and recall, and the results show that. But if we really want to apply this to the real world, we'll need to think further about what really matters to end-users and other stakeholders. 


```python

```
