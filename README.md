# MSL-SignLanguageRecognizer
This project’s main aim is to develop Real Time Malaysian Sign Language Interpretation Using Hand Gesture platform that allows people to engage and learn sign language. This project will make use of Machine learning Algorithms and computer vision technology (STREAMLIT) to create the application.

Web Application Interface using Streamlit
![image](https://github.com/JojiWills/MSL-SignLanguageRecognizer/assets/109582424/8cbdf6b5-8b39-464d-a29b-023e785ad8ad)

Approach:

1. Google's MediaPipe hand-tracking technology to recognize and translate Malaysian Sign Language (MSL) hand gestures.
   ![image](https://github.com/JojiWills/MSL-SignLanguageRecognizer/assets/109582424/ee81d4d2-cf77-49c3-a764-95257815b448)

2. The capabilities of machine learning and computer vision technologies.
   ![image](https://github.com/JojiWills/MSL-SignLanguageRecognizer/assets/109582424/807e23b6-8e0d-43cd-a85a-dbf58da85b9e)


Machine Learning Model:
**Support Vector Machine**
# In[1]:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle

# In[2]:
df = pd.read_csv('dataset.csv')
df.columns = [i for i in range(df.shape[1])]
df

# In[3]:
df = df.rename(columns={63: 'Output'})
df

# In[4]:
X = df.iloc[:, :-1]
print("Features shape =", X.shape)

Y = df.iloc[:, -1]
print("Labels shape =", Y.shape)

# In[5]:
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
svm = SVC(C=10, gamma=0.1, kernel='rbf')
svm.fit(x_train, y_train)

# In[6]:
y_pred = svm.predict(x_test)
y_pred

# In[7]:
cf_matrix = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
precision = precision_score(y_test, y_pred, average='micro')
f1, recall, precision

# In[8]:
labels = sorted(list(set(df['Output'])))
labels = [x.upper() for x in labels]

fig, ax = plt.subplots(figsize=(12, 12))

ax.set_title("Confusion Matrix - Malaysian Sign Language")

maping = sns.heatmap(cf_matrix, 
                     annot=True,
                     cmap = plt.cm.Blues, 
                     linewidths=.2,
                     xticklabels=labels,
                     yticklabels=labels, vmax=8,
                     fmt='g',
                     ax=ax
                    )
maping

# In[9]:

with open('model.pkl','wb') as f:
    pickle.dump(svm,f)

![image](https://github.com/JojiWills/MSL-SignLanguageRecognizer/assets/109582424/49f369ab-fe3e-49c2-a3b2-b14e12bf93c1)


**MSL-recognizer**
![Futuristic Technology Conference Presentation](https://github.com/JojiWills/MSL-SignLanguageRecognizer/assets/109582424/832fa047-f82d-47bc-95d9-13f822ead52c)


