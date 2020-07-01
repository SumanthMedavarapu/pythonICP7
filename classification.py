from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train_tfidf,twenty_train.target)
predicted = clf.predict(X_test_tfidf)
predicted1 = svclassifier.predict(X_test_tfidf)
score = metrics.accuracy_score(twenty_test.target, predicted)
score1 = metrics.accuracy_score(twenty_test.target, predicted1)
print("score with multinomialNB"+score)
print("score after using svm"+score1)
#Here we used bigram
tfidf_Vect1 = TfidfVectorizer(ngram_range=(2,2))
X_train_tfidf1 = tfidf_Vect1.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
clf1 = MultinomialNB()
clf1.fit(X_train_tfidf1, twenty_train.target)


X_test_tfidf1 = tfidf_Vect.transform(twenty_test.data)
predicted3 = clf1.predict(X_test_tfidf1)

score3 = metrics.accuracy_score(twenty_test.target, predicted3)
print("score after using bigram"+score3)
#stop words english
tfidf_Vect2 = TfidfVectorizer(stop_words='english')
X_train_tfidf2 = tfidf_Vect2.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
clf2 = MultinomialNB()
clf2.fit(X_train_tfidf2, twenty_train.target)


X_test_tfidf2 = tfidf_Vect.transform(twenty_test.data)
predicted2 = clf2.predict(X_test_tfidf2)

score2 = metrics.accuracy_score(twenty_test.target, predicted2)

print("score after setting argument as stop words"+score2)