#importing the libraries
import nltk,re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pandas as pd

#import the pandas cv
messages = pd.read_csv(r"D:\SAAP\StockSence\archive SS\all-data.csv",encoding="Latin-1", names=['sentiment','news'], header=None)

#using the stemming*
snow_ball=SnowballStemmer('english')
keep = {'not', 'no', 'nor', 'up', 'down', 'over', 'under', 'above', 'below'}
stop_words=set(stopwords.words('english')) - keep
corpus=[]

# Cleaning the data
for i in range(0,len(messages)):
   review = re.sub('[^a-zA-Z]'," ", messages['news'][i])
   review=review.lower()
   review=review.split()
   review=[snow_ball.stem(word) for word in review if word not in stop_words]
   review = " ".join(review)
   corpus.append(review)


#train & test
from sklearn.model_selection import train_test_split
y=messages['sentiment']
X_train, X_test, y_train, y_test= train_test_split(
   corpus, y, test_size=0.2, random_state=42, stratify=y
)

#using pipeline for more accuracy*

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipe=Pipeline([
   ('tfidf',TfidfVectorizer(sublinear_tf=True,min_df=2)),
   ('clf',LogisticRegression(max_iter=1000,class_weight="balanced")),
])


#grid search
from sklearn.model_selection import GridSearchCV
params={
   'tfidf__ngram_range':[(1,1),(1,2),(1,3)],
   'clf__C':[0.5,1,5,10],
}
grid=GridSearchCV(pipe,params,scoring="f1_macro",cv=5)
grid.fit(X_train,y_train)

print("Best Settings:",grid.best_params_)
print("Best CV Macro-F1",grid.best_score_)


#predict & evaluate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = grid.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))







'''
# Word Embedding usage 
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(max_features=10000,ngram_range=(1,2),min_df=2,sublinear_tf=True)
X = tfidf.fit_transform(corpus).toarray() #text → numbers array

#train the classifier
from sklearn.naive_bayes import MultinomialNB
model= MultinomialNB()
model.fit(X_train, y_train) #numbers → learn

'''