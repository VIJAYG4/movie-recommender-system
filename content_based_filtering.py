import pandas as pd 
import numpy as np 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#---------------------------------------------------------------------
#read csv file
movie_df = pd.read_csv('movie_dataset.csv')
print("column names are ", movie_df.columns)
print("length of movie df is", len(movie_df))
print(movie_df.head())
#--------------------------------------------------------------------
#helper functions
def get_index_from_title(title): 
    index_df = movie_df[movie_df['title']==title]
    return index_df['index']
      

def get_title_from_index(index):
    title_df = movie_df[movie_df['index']==index]
    return title_df['title'].values[0]
#---------------------------------------------------------------------

#step 1: select relevant features
features = ['genres', 'cast', 'director', 'keywords']

for feature in features:
    movie_df[feature] = movie_df[feature].fillna('')

def combineFeatures(row):
    try:
        return row['genres']+" "+row['cast']+" "+row["director"]+" "+row['keywords']
    except:
        print("error is", row)

#combine all features into one
movie_df['combined_feature']=movie_df.apply(combineFeatures,axis=1)
#-------------------------------------------------------------------------------------------

#step 2: calculate count matrix for combined_feature column
cv = CountVectorizer()
count_mat = cv.fit_transform(movie_df['combined_feature'])
#print("shape of count matrix is", count_mat.shape) (num_movies,vocab)

#step 3: calculate cosine similarity of movies
similarity_mat = cosine_similarity(count_mat)
#print("shape of similarity matrix is", similarity_mat.shape) (num_movies,num_movies)

#step 4:
user_like_movie='Avatar'
#get movie index
index = get_index_from_title(user_like_movie)

#get descending order of similarity score indexes
movie_recommendations_indexes = np.flip(np.argsort(similarity_mat[index]))[0]
#print("movie recommendations indexes are", movie_recommendations_indexes[:10])

#step 5: convert movie index to titles and display the top 10 recommendations
print("Your Movie Recommendations are")
cnt=0
for movie_index in movie_recommendations_indexes: 
    movie_name = get_title_from_index(movie_index)
    print(movie_name)
    cnt+=1
    if cnt>10:
        break 
#--------------------------------------------------------------------------------------------