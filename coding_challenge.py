import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import LancasterStemmer
import string
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


nltk.download('stopwords')
nltk.download('punkt')

# Data Preprocessing
df = pd.read_csv("/content/drive/MyDrive/data/video_label/Eluvio_DS_Challenge.csv")

def text_cleaning(df):
    for i, text in enumerate(df['title']):
        # remove punctuation
        s = text.translate(str.maketrans('', '', string.punctuation))
        word_tokens = word_tokenize(s)
        # remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_s = [w for w in word_tokens if w not in stop_words]
        filtered_s = ' '.join(filtered_s)
        # stemming
        my_stemmer = LancasterStemmer()
        stemmed = my_stemmer.stem(filtered_s)
        df.at[i, 'title'] = stemmed
        if (i%100000 == 0): print(i,"completed.")

# text cleaning
text_cleaning(df)

# create keyword
for i,value in enumerate(df['title']):
  keyword = value + " " + str(df['over_18'][i]) + " " + df['author'][i]
  df.at[i, 'keyword'] = keyword


X = np.array(df["title"])
y = np.array(df["up_votes"])

# tokenize and vectorize
def token_vectorize(X):
  tokenizer = Tokenizer(num_words=50000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
  tokenizer.fit_on_texts(X)
  word_index = tokenizer.word_index
  print('Found %s unique tokens.' % len(word_index))
  X = tokenizer.texts_to_sequences(X)
  X = pad_sequences(X)
  print('Shape of data tensor:', X.shape)
  return X

X = token_vectorize(X)

# split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Votes Prediction
# 1a. LSTM

# normalize y
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
normalized_y_train = preprocessing.normalize(y_train)
normalized_y_test = preprocessing.normalize(y_test)

# define lstm model
model = Sequential()
model.add(Embedding(500000, 100, input_length=X_train.shape[1]))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# lstm model training
model.fit(X_train, normalized_y_train, epochs=2, batch_size=128)

# prediction
results_lstm = model.predict(X_test)

# mean accuracy
score_lstm = model.evaluate(X_test,normalized_y_test)

print("lstm accuracy:", (score_lstm[1])*100, "%")


# 1b. Machine Learning Algorithms
# linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_lr_pred = lr.predict(X_test)

# K-NN Regressor
knn_regressor=KNeighborsRegressor(n_neighbors = 5)
knn_model=knn_regressor.fit(X_train,y_train)
y_knn_pred=knn_model.predict(X_test)

# Random Forest Regressor
rf = RandomForestRegressor()
rf_model = rf.fit(X_train, y_train)
y_rf_pred=rf_model.predict(X_test)

# create predictive table
y_t = []
for val in list(y_test):
  num = str(val)
  num = num[1:-1]
  y_t.append(int(num))

predict_table = pd.DataFrame(columns=['y_lr_pred','y_knn_pred','y_rf_pred', 'average','y_test'])
predict_table['y_test'] = y_t
predict_table['y_lr_pred'] = y_lr_pred
predict_table['y_knn_pred'] = y_knn_pred
predict_table['y_rf_pred'] = y_rf_pred
  
for i, val in enumerate(predict_table['y_lr_pred']):
  predict_table.at[i,'average'] = (int(val) + int(predict_table['y_knn_pred'][i]) + int(predict_table['y_rf_pred'][i]))/3


# 2. Recommend similar video

# vectorize the keywords
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vector = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vector.fit_transform(df['keyword'][:10000]) 
# limit data size to avoid crashing due to high computional power in calculating cosine similarity

# cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# make indices
indices = pd.Series(df.index,index=df['title']).drop_duplicates()

def single_text_cleaning(text):
  # remove punctuation
  s = text.translate(str.maketrans('', '', string.punctuation))
  word_tokens = word_tokenize(s)
  # remove stopwords
  stop_words = set(stopwords.words('english'))
  filtered_s = [w for w in word_tokens if w not in stop_words]
  filtered_s = ' '.join(filtered_s)
  # stemming
  my_stemmer = LancasterStemmer()
  stemmed = my_stemmer.stem(filtered_s)
  return stemmed

# content-based recommender
def get_recommendation(original_title,cosine_sim=cosine_sim,df=df):
    title = single_text_cleaning(original_title)
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    video_indices = [i[0] for i in sim_scores]
    
    recommendations = df.iloc[video_indices][['title','up_votes','down_votes','over_18','author']]
    recommendations = recommendations.sort_values('up_votes', ascending=False)
    return recommendations

# examples: (title) US presses Egypt on Gaza border
rec_list = get_recommendation("US presses Egypt on Gaza border")
