#### ---- importing libraries ---- ####
import pandas as pd
import string
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.svm import LinearSVC

#### ---- loading data into dataframe ---- ####

# load training data into Pandas DataFrame
df = pd.read_json('data/train/train.json')
test = pd.read_json('data/test/test.json')

# merging data columns into one
df['all_features'] = df['abstract'] + ' ' +  df['title'] + ' ' + df['year'].apply(str) + ' ' + df['abstract']
test['all_features'] = test['abstract'] + ' ' +  test['title'] + ' ' + test['year'].apply(str) + ' ' + test['abstract']

#### ---- cleaning the data ---- ####
def cleaning_data(df_name, colum_name):
    # Lowercase
    df_name[colum_name] = df_name[colum_name].str.lower()

    # remove punctuation
    df_name[colum_name] = df_name[colum_name].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))

    # stemming
    stemmer = PorterStemmer()
    def stem_words(text):
        return " ".join([stemmer.stem(word) for word in text.split()])
    df_name[colum_name] = df_name[colum_name].apply(lambda x: stem_words(x))

    # lemmatization
    lemmatizer = WordNetLemmatizer()
    def lemmatize_words(text):
        return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    df_name[colum_name] = df_name[colum_name].apply(lambda text: lemmatize_words(text))

    # remove extra spaces
    df_name[colum_name] = df_name[colum_name].apply(lambda x: re.sub(' +', ' ', x))
    
# clean test and train data    
cleaning_data(df, 'all_features')
cleaning_data(test, 'all_features')
        
#### ---- feature engineering  ---- ####

# TF_IDF reflects the importance of a word or phrase in a document
vectorizer = TfidfVectorizer(min_df=2,ngram_range=(1,2), sublinear_tf=True, token_pattern=r'([a-zA-Z0-9]{1,})')

# returns raw documents into a matrix of numerical values
trainVector = vectorizer.fit_transform(df["all_features"])
testVector = vectorizer.transform(test["all_features"])

# Label encoding for decision classes (author IDs)
le = LabelEncoder()
Y_label = le.fit_transform(df['authorId'])

#### ---- prediction model  ---- ####

# linear support vector classifier from scikit-learn machine learning library
svm = LinearSVC(C=40)

# fitting the model to training data
svm_fit = svm.fit(trainVector, Y_label)

# generating predictions for test data
predictions_svm = svm_fit.predict(testVector)

# adding the predictions to our dataframe
authorId = le.inverse_transform(predictions_svm)
test["authorId"] = authorId
test["authorId"] = test["authorId"].apply(str)

# exporting our predictions file to .json format
test.to_json("predicted.json", orient= "records")