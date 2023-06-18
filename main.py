#!/usr/bin/env python
# coding: utf-8

# # Fake news Detection
# #### Fake news on different platforms is spreading widely and is a matter of serious concern, as it causes social wars and permanent breakage of the bonds established among people. A lot of research is already going on focused on the classification of fake news.
# 
# Code: https://www.kaggle.com/rodolfoluna/fake-news-detector
# 

# ## Steps to be followed
# #### 1- Importing Libraries and Datasets
# #### 2-Data Preprocessing
# #### 3-Preprocessing and analysis of News column
# #### 4-Converting text into Vectors
# #### 5-Model training, Evaluation, and Prediction

# ### Importing required library
# Here we are going to importing some of the required library.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


# ### Inserting fake and real dataset

# In[2]:


df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")


# ### Data Preprocessing

# In[4]:


df_fake.isna().sum() #NaN
df_true.isna().sum() #NaN


# In[5]:


df_fake.describe()


# In[6]:


df_true.describe()


# In[7]:


df_fake.head(5)


# In[8]:


df_true.head(5)


# Inserting a column called "class" for fake and real news dataset to categories fake and true news. 

# In[9]:


df_fake["target"] = 'fake'
df_true["target"] = 'true'


# Merging the main fake and true dataframe

# In[17]:


df_marge = pd.concat([df_fake, df_true], axis =0 )
df_marge.head(5)


# # Preprocessing and analysis of News column

# ### Data cleaning and preparation

# In[18]:


df_marge.columns


# #### "title" and "date" columns is not required for detecting the fake news, so I am going to drop the columns.

# In[19]:


df = df_marge.drop(["title","date"], axis = 1)


# #### Randomly shuffling the dataframe 

# In[21]:


# Shuffle the data
from sklearn.utils import shuffle
df = shuffle(df)
df = df.reset_index(drop=True)


# In[26]:


df.head()


# #### Creating a function to convert the text in lowercase, remove the extra space, special chr., ulr, punctuation and links .

# In[27]:


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    all_list = [char for char in text if char not in string.punctuation]
    text = ''.join(all_list)

    return text


# In[28]:


df["text"] = df["text"].apply(wordopt)


# In[29]:


# Check
df.head()


# In[30]:


# Removing stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[31]:


# Check
df.head()


# ### Basic data exploration
# 

# In[32]:


# How many articles per subject?
print(df_marge.groupby(['subject'])['text'].count())
df_marge.groupby(['subject'])['text'].count().plot(kind="bar")
plt.show()


# In[33]:


# How many fake and real articles?
print(df.groupby(['target'])['text'].count())
df.groupby(['target'])['text'].count().plot(kind="bar")
plt.show()


# In[34]:


# Word cloud for fake news
from wordcloud import WordCloud

fake_data = df[df["target"] == "fake"]
all_words = ' '.join([text for text in fake_data.text])

wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[35]:


# Word cloud for real news
from wordcloud import WordCloud

real_data = df[df["target"] == "true"]
all_words = ' '.join([text for text in real_data.text])

wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[36]:


# Most frequent words counter
from nltk import tokenize

token_space = tokenize.WhitespaceTokenizer()

def counter(text, column_text, quantity):
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                   "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'purple')
    ax.set(ylabel = "Count")
    plt.xticks(rotation='vertical')
    plt.show()


# In[37]:


# Most frequent words in fake news
counter(df[df["target"] == "fake"], "text", 20)


# In[38]:


# Most frequent words in real news
counter(df[df["target"] == "true"], "text", 20)


# ### Modeling

# In[39]:


# Function to plot the confusion matrix (code from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)
from sklearn import metrics
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# #### Defining dependent and independent variable as x and y

# In[40]:


x = df["text"]
y = df["target"]


# #### Splitting the dataset into training set and testing set. 

# In[41]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# #### Convert text to vectors

# In[42]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[43]:


vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# # Model training, Evaluation, and Prediction

# ### 1. Logistic Regression

# In[44]:


dct = dict()
TM=dict()
from sklearn.linear_model import LogisticRegression
import time


# In[45]:


t0=time.time()
LR = LogisticRegression()
LR.fit(xv_train,y_train)
pred_lr=LR.predict(xv_test)

t1=time.time()
print("accuracy: {}%".format(round(LR.score(xv_test, y_test)*100,2)))
dct['Logistic Regression'] = round(accuracy_score(y_test, pred_lr)*100,2)
TM['Logistic Regression'] = round(t1-t0)


# In[48]:


cm = metrics.confusion_matrix(y_test, pred_lr)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])
print(classification_report(y_test, pred_lr))
print("The time take",(t1-t0))


# ### 2. Decision Tree Classification

# In[49]:


from sklearn.tree import DecisionTreeClassifier
import time


# In[50]:


t0=time.time()
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
t1=time.time()
print("accuracy: {}%".format(round(DT.score(xv_test, y_test)*100,2)))


# In[51]:


DT = metrics.confusion_matrix(y_test, pred_dt)
plot_confusion_matrix(DT, classes=['Fake', 'Real'])
print(classification_report(y_test, pred_dt))
print("The time take",(t1-t0))
dct['Decision Tree'] = round(accuracy_score(y_test, pred_dt)*100,2)
TM['Decision Tree'] = round(t1-t0)


# ### 3. Gradient Boosting Classifier

# In[52]:


from sklearn.ensemble import GradientBoostingClassifier
import time


# In[53]:


t0=time.time()
GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)
pred_gbc = GBC.predict(xv_test)

t1=time.time()
print("accuracy: {}%".format(round(GBC.score(xv_test, y_test)*100,2)))


# In[54]:


GB = metrics.confusion_matrix(y_test, pred_gbc)


plot_confusion_matrix(GB, classes=['Fake', 'Real'])
print(classification_report(y_test, pred_gbc))
print("The time Take",(t1-t0))
dct['Gradient Boosting'] = round(accuracy_score(y_test, pred_gbc)*100,2)
TM['Gradient Boosting'] = round(t1-t0)


# ### 4. Random Forest Classifier

# In[55]:


from sklearn.ensemble import RandomForestClassifier
import time


# In[56]:


t0=time.time()
RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)
pred_rfc = RFC.predict(xv_test)
t1=time.time()
print("accuracy: {}%".format(round(RFC.score(xv_test, y_test)*100,2)))


# In[57]:


RF = metrics.confusion_matrix(y_test, pred_rfc)

plot_confusion_matrix(RF, classes=['Fake', 'Real'])
print(classification_report(y_test, pred_rfc))
print("The time Take",(t1-t0))
dct['Random Forest'] = round(accuracy_score(y_test, pred_rfc)*100,2)
TM['Random Forest'] = round(t1-t0)


# ### 5. SVM

# In[58]:


from sklearn import svm
import time


# In[59]:


t0=time.time()
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel
clf.fit(xv_train, y_train)
pred_clf = clf.predict(xv_test)
t1=time.time()
print("accuracy: {}%".format(round(clf.score(xv_test, y_test)*100,2)))


# In[60]:


SV = metrics.confusion_matrix(y_test, pred_clf)

plot_confusion_matrix(SV, classes=['Fake', 'Real'])
print(classification_report(y_test, pred_clf))
print("The time Take",(t1-t0))
dct['SVM'] = round(accuracy_score(y_test, pred_clf)*100,2)
TM['SVM'] = round(t1-t0)


# ### 6.Naive Bayes
# 

# In[61]:


from sklearn.naive_bayes import MultinomialNB
import time


# In[62]:


#Create a NB Classifier
t0=time.time()
NB_classifier = MultinomialNB()
NB_classifier.fit(xv_train, y_train)
pred_NB = NB_classifier.predict(xv_test)

t1=time.time()
print("accuracy: {}%".format(round(NB_classifier.score(xv_test, y_test)*100,2)))


# In[63]:


NB = metrics.confusion_matrix(y_test, pred_NB)

plot_confusion_matrix(NB, classes=['Fake', 'Real'])
print(classification_report(y_test, pred_NB))
print("The time Take",(t1-t0))
dct['Naive Bayes'] = round(accuracy_score(y_test, pred_NB)*100,2)
TM['Naive Bayes'] = round(t1-t0)


# ## Comparing Different Models

# ### Compare between accuracy

# In[64]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8,7))
plt.bar(list(dct.keys()),list(dct.values()))
plt.xticks(rotation='vertical')
plt.ylim(90,100)
plt.yticks((91, 92, 93, 94, 95, 96, 97, 98, 99, 100))


# ### Compare between Time

# In[66]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8,7))
plt.bar(list(TM.keys()),list(TM.values()))
plt.xticks(rotation='vertical')

