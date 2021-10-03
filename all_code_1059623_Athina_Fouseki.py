from elasticsearch import Elasticsearch
import pandas as pd
import prettytable
import math
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from statistics import mean
import string
import numpy as np
## for data
import pandas as pd
## for processing
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from nltk import word_tokenize
from nltk.corpus import stopwords
## for deep learning
from sklearn import model_selection
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant

# Connect to the elastic cluster
# es=Elasticsearch([{'host':'localhost','port':9200}], http_auth=('elastic', '123123123#'))

es=Elasticsearch([{'host':'localhost','port':9200}])

# Delete index
#es.indices.delete(index="cinema")

#Create the cinema index
#es.indices.create(index="cinema")
print(es.indices.exists(index="cinema")) #returns true

df_mv = pd.read_csv('movies.csv')
df_mv['genres']=df_mv['genres'].str.split('\|')
df_mv['title']=df_mv['title'].str.split('\(').str[0]
print(df_mv['genres'].head(10))

df_rat = pd.read_csv('ratings.csv')
df_rat.drop(['timestamp'], axis = 1, inplace=True)
df_rat["rating"] = pd.to_numeric(df_rat["rating"], downcast="float")

#print(df_mv.head(10))

#for index, row in df_mv.iterrows():
#    temp = {"movieid": row['movieId'], "title": row['title'], "genres": row['genres']}
#    es.index(index="cinema",  id=index+1, body=temp)
    #print(row['movieId'], row['title'], row['genres'], end='\t')


# ~~~~~~START: NEURAL NETWORK TRAINING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# I want to find all the movies that have ratings
df = pd.DataFrame(df_rat['movieId'].unique(), columns=['unique_movies']) # contains the movies with ratings - going to use it after the word embedding phase
# print(df.head(20))

#we need to ONE HOT ENCODE the different genres - LATER
#mlb = MultiLabelBinarizer()
#df_mv = df_mv.join(pd.DataFrame(mlb.fit_transform(df_mv.pop('genres')),
#                          columns=mlb.classes_,
#                          index=df_mv.index))
# print(df_mv.head(20).to_string())

#we are going to use all the rated movies in rating.csv for the training and testing of the neural network
mvs_nn = df_rat.merge(df_mv[['title', 'movieId']], on='movieId', how='inner')

print(mvs_nn.head(50).to_string())

#doing the glove stuff 
def remove_punct(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)

mvs_nn["title"] = mvs_nn.title.map(lambda x: remove_punct(x))
stop = set(stopwords.words("english"))


def remove_stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(text)

mvs_nn["title"] = mvs_nn["title"].map(remove_stopwords)

def create_corpus_tk(mvs_nn):
    corpus = []
    for text in mvs_nn:
        words = [word.lower() for word in word_tokenize(text)]
        corpus.append(words)
    return corpus

corpus = create_corpus_tk(mvs_nn["title"])

num_words = len(corpus)# Size of vocabulary obtained when preprocessing text data 

train, test = model_selection.train_test_split(mvs_nn, test_size=0.25)

max_len = 50
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(train["title"])

train_sequences = tokenizer.texts_to_sequences(train["title"])

train_padded = pad_sequences(
    train_sequences, maxlen=max_len, truncating="post", padding="post"
)
test_sequences = tokenizer.texts_to_sequences(test["title"])
test_padded = pad_sequences(
    test_sequences, maxlen=max_len, padding="post", truncating="post"
)
word_index = tokenizer.word_index


embedding_dict = {}
with open('glove.6B.100d.txt', encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:], "float32")
        embedding_dict[word] = vectors
f.close()

num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, 100))

for word, i in word_index.items():
    if i < num_words:
        emb_vec = embedding_dict.get(word)
        if emb_vec is not None:
            embedding_matrix[i] = emb_vec

print(train_padded)
print(train_padded.shape)
print(train["rating"].shape)
print(train["rating"])
print(train["userId"])


# attempt to do the NEURAL NETWORK stuff
num_users = 1 # single user every time
title_input = keras.Input(shape=(None,), name="title")  # Variable-length sequence of ints
user_input = keras.Input(shape=(num_users, ), name="user")

title_features = layers.Embedding(num_words, 100, embeddings_initializer=Constant(embedding_matrix),
                                  input_length=max_len, trainable=False )(title_input) #EMBEDDING 

# Reduce sequence of embedded words in the title into a single 128-dimensional vector
title_features = layers.LSTM(128, dropout=0.2, activation='sigmoid')(title_features)

# Merge all available features into a single large vector via concatenation
x = layers.concatenate([title_features, user_input])

# Stick a logistic regression for priority prediction on top of the features
rating_pred = layers.Dense(1, activation='linear', name="rating")(x)

# Instantiate an end-to-end model predicting both priority and department
model = keras.Model(
    inputs=[title_input, user_input],
    outputs=[rating_pred]
)


model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mse', 'mae', 'mape', 'cosine_proximity']
)


keras.utils.plot_model(model, "multi_input_and_output_model2.png", show_shapes=True) # PLOTTING THE MODEL
# Dummy input data
title_data = train_padded
user_data = train['userId']

# Dummy target data
rating_targets = train['rating']

model.fit(
    {"title": title_data, "user": user_data},
    {"rating": rating_targets},
    epochs=1,
    batch_size=20,
    verbose = 1
)


# evaluate the model
loss, mse, mae, mape, cosine = model.evaluate(x={'title': test_padded, 'user': test['userId']},
                            y={'rating': test["rating"]}, verbose=1)

print('mse: %f' % mse)
print('mae: %f' % mae)
print('mape: %f' % mape)
print('cosine: %f' % cosine)

# ~~~~~~END: NEURAL NETWORK TRAINING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~Q 3~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# calculate the average of ratings in every movie genres for every user
df_new = df_rat.join(df_mv['genres'], how='inner',
                     on='movieId')  # creating a new dataframe that contains the genres of the movie for every rated movie
df_new = df_new.explode('genres')  # exploding the genres column in order to get the unique genres
unq_gens = df_new['genres'].unique()  # list of unique genres

# if we print the unq_gens list, we are going to see that some movies don't have
# genres. We're going to drop these movies from the dataset.
df_new.drop(df_new[df_new['genres'] == '(no genres listed)'].index, inplace=True)
unq_gens = df_new['genres'].unique()  # list of unique genres

unq_users = df_new['userId'].unique()  # list of unique users

df_avg = pd.DataFrame(columns=['userId', 'genre', 'avg_rating'])  # new dataframe
df_avg2_cols = ['userId']
df_avg2_cols.extend(unq_gens)
df_avg2 = pd.DataFrame( columns=df_avg2_cols)  # second new dataframe to use for kmeans model fitting - contains the avg ratings per user for each genre


i = 0
for user in unq_users:
    df_avg2 = df_avg2.append({'userId': str(user)}, ignore_index=True)
    for genre in unq_gens:
        avg = round(df_new[(df_new['genres'] == genre) & (df_new['userId'] == user)]['rating'].mean(axis=0), 2)
        if math.isnan(avg):
            new_row = {'userId': user, 'genre': genre, 'avg_rating': 'no rating'}
            # append row to the dataframe
            df_avg = df_avg.append(new_row, ignore_index=True)
            df_avg2.at[i, genre] = float('nan')
        else:
            new_row = {'userId': user, 'genre': genre, 'avg_rating': avg}
            # append row to the dataframe
            df_avg = df_avg.append(new_row, ignore_index=True)
            df_avg2.at[i, genre] = avg
    i += 1

for col in df_avg2.columns:
    if col != 'userId':
        res = (df_avg2[col].isna()).sum()
        percnt = int(res) / int(len(df_avg2))
        


for col in df_avg2.columns:
    
    if col != 'userId':
        df_avg2[col].fillna(float(df_avg2[col].mean()), inplace=True)


# now i have to use df_avg2 to fit the kmeans model
X = df_avg2.drop(['userId'], axis=1).values
wcss = []

for i in range(1, 11):  # stuff for the elbow method to calculate the best k
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

clstrs = int(input("Give number of clusters: "))
kmeans = KMeans(n_clusters=clstrs, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)
data = {"Cluster": pred_y}
df_pred = pd.DataFrame(data)

df_avg_pred = df_avg2.join(df_pred) #contains the avg ratings per user for each genre and the cluster each user belongs to

user_per_cluster_list = []  # will contain lists of the users per cluster
for i in range(0, clstrs):
    temp_list = df_avg_pred[df_avg_pred['Cluster'] == i]['userId'].to_list()
    temp_list = list(map(int, temp_list))
    user_per_cluster_list.append(temp_list)


# stuff for calculating an avg rating for every movie per cluster - it's enough to run it once, just to get the csv files
    '''
movies_with_rat = pd.DataFrame(columns = ['movieId', 'rating', 'cluster']) #new dataframe that contain the movies with ratings
movies_no_rat = pd.DataFrame(columns = ['movieId', 'rating', 'cluster']) #new dataframe that contain the movies with zero ratings


for movie in df_mv['movieId']:
    for cluster in range(0,clstrs):
        ratings_of_wanted_genres = []
        temp_rt = df_rat[df_rat['movieId'] == movie]

        if temp_rt.empty:
            gens = df_mv.iloc[df_mv[df_mv['movieId']==movie].index.item()]['genres'] #keeping the genres that the movie belongs to in a list -extra to_list probs not needed
            temp_avg = df_avg_pred[df_avg_pred['Cluster'] == cluster] #temp dataframe that contains the data for the AVG rating of the movie
            try:
                temp_avg = temp_avg[gens]

                for colu in gens:
                    ratings_of_wanted_genres.append(float(temp_avg[colu].mean())) #list of the average ratings of the wanted genres in a particular cluster
                average_rat = mean(ratings_of_wanted_genres) #calculating the mean of all the ratings the list contains
                movies_no_rat = movies_no_rat.append({'movieId': str(int(movie)),'rating': average_rat, 'cluster': str(int(cluster))},ignore_index=True ) #adding the data to the dataframe
            except:
                movies_no_rat = movies_no_rat.append({'movieId': str(int(movie)),'rating': 2.5, 'cluster': str(int(cluster))}, ignore_index=True) #adding the data to the dataframe            
        else:

            m = df_rat[(df_rat['movieId'] == movie) & (df_rat['userId'].isin(user_per_cluster_list[cluster]))]['rating'].mean()
            #print("m: {}".format(m))
            if math.isnan(m):
                gens = df_mv.iloc[df_mv[df_mv['movieId']==movie].index.item()]['genres'] #keeping the genres that the movie belongs to in a list -extra to_list probs not needed
                temp_avg = df_avg_pred[df_avg_pred['Cluster'] == cluster] #temp dataframe that contains the data for the AVG rating of the movie

                try:
                    temp_avg = temp_avg[gens]
                    for colu in gens:
                        ratings_of_wanted_genres.append(float(temp_avg[colu].mean())) #list of the average ratings of the wanted genres in a particular cluster

                    average_rat = mean(ratings_of_wanted_genres) #calculating the mean of all the ratings the list contains
                    movies_with_rat = movies_with_rat.append({'movieId': str(int(movie)),'rating': average_rat, 'cluster': str(int(cluster))},ignore_index=True ) #adding the data to the dataframe
                except:
                    movies_with_rat = movies_with_rat.append({'movieId': str(int(movie)),'rating': 2.5, 'cluster': str(int(cluster))}, ignore_index=True) #adding the data to the dataframe            
            else:
                movies_with_rat = movies_with_rat.append({'movieId': str(int(movie)),'rating': m, 'cluster': str(int(cluster))}, ignore_index=True)


#print(movies_with_rat.head(30).to_string())
#print('\n\n')
#print(movies_no_rat.head(30).to_string())

movies_with_rat.to_csv ('movies_with_rat.csv', index = False, header=True) #passing all the importan info in a csv file
movies_no_rat.to_csv ('movies_no_rat.csv', index = False, header=True) #passing all the importan info in a csv file
'''
# ------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------

# ~~~~~SEARCH STARTS HERE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
user_log=input("Who are you? :") #asking for the user's ID

while True:
    mov={} #storing the result from elasticSearch
    mov2={} #storing the result from elasticSearch

    user_stuff=[]

    final= prettytable.PrettyTable(["Movies To Watch", "BM25 Score","User's Score","Average Rating", "My Metric"])
    final2= prettytable.PrettyTable(["Movies To Watch", "My Metric","User's Score","Average Rating","BM25 Score" ])
    final3= prettytable.PrettyTable(["Movies To Watch", "My 2nd Metric","User's Score","Average Rating","BM25 Score" ])

    titl=input("Give a movie title: ") #the user gives the title of the movie they want to search for

    df_for_pred = pd.DataFrame(columns=['user', 'title'])

    res = es.search(index="cinema", body={"query": {"match": {'title':titl}}}, size=20)

    for hit in res['hits']['hits']: #printing movie title and genres in "pairs"
        df_for_pred = df_for_pred.append({'user': user_log, 'title': hit['_source']['title']}, ignore_index=True)

    df_for_pred['rating'] = ''
    df_for_pred["user"] = pd.to_numeric(df_for_pred["user"])
    df_for_pred['title'] = df_for_pred['title'].astype(str)
    # ~~~~~~START: PREDICT USING THE NEURAL NETWORK~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    df_for_pred['title'] = df_for_pred['title'].apply(str)
    df_for_pred['title'] = df_for_pred.title.map(lambda x: remove_punct(x))
    df_for_pred['title'] = df_for_pred['title'].map(remove_stopwords)

    corpus2 = create_corpus_tk(df_for_pred['title'])

    num_words = len(corpus2)

    max_len = 50
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(df_for_pred['title'])

    train_sequences = tokenizer.texts_to_sequences(df_for_pred['title'])
    to_pred = pad_sequences(train_sequences, maxlen=max_len, truncating="post", padding="post")

    word_index = tokenizer.word_index

    embedding_dict2 = {}
    with open('glove.6B.100d.txt', encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], "float32")
            embedding_dict2[word] = vectors
    f.close()

    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, 100))

    for word, i in word_index.items():
        if i < num_words:
            emb_vec = embedding_dict2.get(word)
            if emb_vec is not None:
                embedding_matrix[i] = emb_vec



    predictions2 = model.predict({"title": to_pred, "user": df_for_pred["user"].values})
    print('Predictions:\n{}'.format(predictions2[:30]))

    df_for_pred['rating'] = predictions2

    # ~~~~~~END: PREDICT USING THE NEURAL NETWORK~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for hit in res['hits']['hits']: #printing movie title and genres in "pairs"
        df_temp = df_rat[(df_rat['movieId'] == int(hit['_id'])) & (df_rat['userId'] == int(user_log))] #dataframe with the user's rating - if not empty
        df_rt = df_rat[df_rat['movieId'] == int(hit['_id'])]
        avg_rat = float(df_rt['rating'].mean(axis = 0)) #average rating of the movie

        #my metric
        if (df_temp.empty == False) and (df_rt.empty == False):
            my_metric = round(0.35*math.exp(hit['_score']) + 0.3*float(df_temp['rating']) + 0.35*avg_rat, 3)
        elif df_temp.empty and (df_rt.empty == False):
            my_metric = round(0.35*math.exp(hit['_score']) + 0.65*avg_rat, 3) #could I give it a little boost?
        else:
            my_metric = round(float(0.35*math.exp(hit['_score']-1))-(math.log(hit['_score'])), 3)

        if df_temp.empty and (df_rt.empty == False):
            mov[hit['_id']]=[hit['_source']['title'], hit['_score'], hit['_source']['genres'], "none", avg_rat, my_metric]#, 0, 0, 0]
        elif df_temp.empty and df_rt.empty:
            mov[hit['_id']]=[hit['_source']['title'], hit['_score'], hit['_source']['genres'], "none", "no ratings", my_metric]
        else:
            mov[hit['_id']]=[hit['_source']['title'], hit['_score'], hit['_source']['genres'], float(df_temp['rating']), avg_rat, my_metric]#, 0, 0, 0]

    for i in mov.items():
        final.add_row([i[1][0], i[1][1], i[1][3], i[1][4], i[1][5]])


    sort_subjects = sorted(mov.items(), key=lambda x: x[1][5], reverse=True)
    print('\n\n\n')
    for i in sort_subjects:
        final2.add_row([i[1][0], i[1][5], i[1][3], i[1][4], i[1][1]])


    #------------------------------------------------------------------------------------------------------------------------------------------------

    movies_with_rat = pd.read_csv('movies_with_rat.csv')
    #movies_no_rat = pd.read_csv('movies_no_rat.csv')

    res2= es.search(index="cinema", body={"query": {"match": {'title':titl}}}, size=20)

    for hit in res2['hits']['hits']: #printing movie title and genres in "pairs"
        df_temp = df_rat[(df_rat['movieId'] == int(hit['_id'])) & (df_rat['userId'] == int(user_log))] #dataframe with the user's rating - if not empty
        df_rt = df_rat[df_rat['movieId'] == int(hit['_id'])]
        avg_rat = float(df_rt['rating'].mean(axis = 0)) #average rating of the movie

        #my metric
        if (df_temp.empty == False) and (df_rt.empty == False):
            my_metric2 = round(0.35*math.exp(hit['_score']) + 0.3*float(df_temp['rating']) + 0.35*avg_rat, 3)
        elif df_temp.empty and (df_rt.empty == False): #user hasn't given a rating to the movie but the movie has ratings
            users_cluster = df_avg_pred[df_avg_pred['userId'] == user_log ]['Cluster'].values
            
            user_rat = movies_with_rat[(movies_with_rat['movieId'] == int(hit['_id'])) & (movies_with_rat['cluster'] == users_cluster[0])]['rating'].values
            user_rat = user_rat[0]
            
            my_metric2 = round(0.35*math.exp(hit['_score']) + 0.3*float(user_rat) + 0.35*avg_rat, 3)
        else:  # the movie has no ratings
            mv = hit['_source']['title']
            mv = str(mv)  
            mv = remove_punct(mv)
            mv = remove_stopwords(mv)
            nn_usr_score = df_for_pred[df_for_pred['title'] == mv]['rating']
            nn_usr_score2 = df_for_pred[df_for_pred['title'] == mv]['rating'].values
            nn_usr_score2 = nn_usr_score2[0]
            print('NN score: \n{}'.format(nn_usr_score))
            print('NN score value: \n{}'.format(nn_usr_score2))
            print("hit title: {}".format(hit['_source']['title']))
            # my_metric2 = round(float(0.35*math.exp(hit['_score']-1))-(math.log(hit['_score'])), 3)
            my_metric2 = round(0.40 * math.exp(hit['_score']) + 0.60 * float(nn_usr_score2), 3)

        if df_temp.empty and (df_rt.empty == False):
            mov2[hit['_id']] = [hit['_source']['title'], hit['_score'], hit['_source']['genres'], user_rat, avg_rat,
                                my_metric2]  
        elif df_temp.empty and df_rt.empty:
            mov2[hit['_id']] = [hit['_source']['title'], hit['_score'], hit['_source']['genres'],
                                '"none | {}"'.format(nn_usr_score2), "no ratings", my_metric2]
        else:
            mov2[hit['_id']] = [hit['_source']['title'], hit['_score'], hit['_source']['genres'], float(df_temp['rating']),
                                avg_rat, my_metric2]  

    sort_subjects2 = sorted(mov2.items(), key=lambda x: x[1][5], reverse=True)
    print('\n\n\n')
    for i in sort_subjects2:
        final3.add_row([i[1][0], i[1][5], i[1][3], i[1][4], i[1][1]])


    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(final)
    print(final2)
    print(final3)
    print(df_for_pred.to_string())

    ans = input('Want to search for another movie? Yes or No? ')
    while (ans != 'yes') and (ans != 'no'):
        ans = input('Please type "yes" or "no". ')

    if ans == 'no':
        break



