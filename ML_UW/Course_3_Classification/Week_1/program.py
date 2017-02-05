import sframe
import math
import numpy
from scipy.special import expit

def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation) 

products = sframe.SFrame('amazon_baby.gl/')
products['review_clean'] = products['review'].apply(remove_punctuation)
products = products[products['rating'] != 3]
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)

train_data, test_data = products.random_split(.8, seed=1)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')

train_matrix = vectorizer.fit_transform(train_data['review_clean'])
test_matrix = vectorizer.transform(test_data['review_clean'])


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(penalty='l2')
sentiment_model=classifier.fit(train_matrix,train_data['sentiment'])

print (sentiment_model.coef_>=0).sum()

sample_test_data = test_data[10:13]
print sample_test_data
sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
scores = sentiment_model.decision_function(sample_test_matrix)
print scores
expit(scores)

test_data['predictions']=sentiment_model.predict_proba(test_matrix)[:,1]
test_data.sort('predictions', ascending=False).print_rows(20,6)
test_data.sort('predictions', ascending=True).print_rows(20,6)

testdata=sentiment_model.predict(test_matrix)
correctly_classified=(testdata==test_data['sentiment']).sum()
accuracy= correctly_classified/float(len(test_data))
print 'Accuracy of sentiment model on test data is %f' % accuracy

## classification accuracy of the sentiment_model on the train_data.
traindata=sentiment_model.predict(train_matrix)
correctly_classified=(traindata==train_data['sentiment']).sum()
accuracy= correctly_classified/float(len(train_data))
print 'Accuracy of sentiment model on train data is %f' % accuracy

## SIMPLE MODEL##

significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']
vectorizer_word_subset = CountVectorizer(vocabulary=significant_words)
train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_data['review_clean'])
test_matrix_word_subset = vectorizer_word_subset.transform(test_data['review_clean'])
simple_model=classifier.fit(train_matrix_word_subset,train_data['sentiment'])
simple_model_coef_table = sframe.SFrame({'word':significant_words,
                                         'coefficient':simple_model.coef_.flatten()})

print simple_model_coef_table



## classification accuracy of the simple_model on the train_data
traindata=simple_model.predict(train_matrix_word_subset)
correctly_classified=(traindata==train_data['sentiment']).sum()
accuracy= correctly_classified/float(len(train_data))
print 'Accuracy of simple model on train data is %f' % accuracy

## classification accuracy of the simple_model on the test_data
testdata=simple_model.predict(test_matrix_word_subset)
correctly_classified=(testdata==test_data['sentiment']).sum()
accuracy= correctly_classified/float(len(test_data))
print 'Accuracy of simple model on test data is %f' % accuracy

## majority class classifier
## reviews with sentiment 1 are high in number based on below
print len(products['sentiment']==1)
print (products['sentiment']==1).sum()
## so majority classifier would classify every review as positive that is into sentiment of 1 (not sentiment of -1)
## so accuracy of majority classifier is as below
print float((products['sentiment']==1).sum())/len(products['sentiment']==1)





