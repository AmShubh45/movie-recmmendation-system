import streamlit as st
import pickle
import pandas as pd
import requests


def fetch_posters(movie_id):
    movie_id = str(movie_id)
    response = requests.get("https://api.themoviedb.org/3/movie/"+movie_id+"?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US")
    data = response.json()
    full_path = "https://image.tmdb.org/t/p/w500" + data['poster_path']
    return full_path



def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = []
    recommended_movies_posters=[]
    for i in movie_list:
        movies_id = movies.iloc[i[0]].id

        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_posters.append(fetch_posters(movies_id))
    return recommended_movies, recommended_movies_posters

st.header('Movies Recommender System')
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))



movie_listi = movies['title'].values
movie_listi = sorted(movie_listi)
selected_movie = st.selectbox(
    "select a movie from the dropdown", movie_listi)

if st.button('Recommend Me'):
    names, posters = recommend(selected_movie)
    #ins=0
    # for i in names:
    #     ins = int(ins) + 1
    #     ins = str(ins)
    #     st.subheader(ins+" ->  "+i)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])