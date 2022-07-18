import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from math import pi
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

st.title('Welcome to the Lesta Song Recommendation App!')



col_names = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
             'liveness', 'valence', 'tempo', 'type', 'id', 'uri', 'track_href', 'analysis_url', 'duration_ms',
             'time_signature', 'genre', 'song_name', 'Unnamed: 0', 'title']
df = pd.read_csv("songs.csv", names = col_names, low_memory=False)
st.write('This application has data on', df.shape[0], 'Spotify songs. Below is a sample of the data:')
st.dataframe(df.head(10))


# pie chart of genre
st.write('The dataset is comprised of songs from several genres. The pie chart below shows the distribution of the songs across the different genres:')
genres = df.genre.value_counts()
pie_chart_labels = ['Underground Rap', 'Dark Trap', 'Hiphop', 'RnB', 'Emo', 'Rap', 'Trap Metal', 'Pop']
pie_chart_values = genres.values
pie_chart_fig = go.Figure(data=[go.Pie(labels=pie_chart_labels, values=pie_chart_values)])
st.plotly_chart(pie_chart_fig)


st.write("The machine learning algorithm uses the following parameters to find similar songs: 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'")

# Combine columns so that we can use them to find similarity between other songs
def get_combined_columns():
        combined_columns = []
        columns = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        for i in range(0, df.shape[0]):
            combo = ""
            for col in columns:
                combo += str(df[col][i])
                combo += ' '
            combined_columns.append(combo)
        return combined_columns


df['combined_columns'] = get_combined_columns()
csm = cosine_similarity(CountVectorizer().fit_transform(df['combined_columns']))

# display option box to choose a song
selected_song = st.selectbox(
    'Which song do you want to find similar songs to?',
    df['song_name']
)
selected_song_data = df[df['song_name'] == selected_song]
selected_song_index = selected_song_data.index.values[0]


def get_song_data(song):
    song_id = song[0]

    data = []
    for label in spoke_labels:
        data.append(df.at[song_id, label])
    return data

# radar chart of selected song
st.write('"', selected_song, '" has characteristics that look like this:')
spoke_labels = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']
values = get_song_data(selected_song_data.index.values)
radar_df = pd.DataFrame(dict(
    r=values,
    theta=spoke_labels
))
fig = px.line_polar(radar_df, r='r', theta='theta', line_close=True, range_r=[0,1])
st.plotly_chart(fig)


similar_songs = enumerate(csm[selected_song_index])
sorted_similar_songs = sorted(similar_songs, key = lambda x:x[1], reverse = True)



def get_song_name(ranked_song):
    song_id = ranked_song[0]

    return df.at[song_id, 'song_name']


# radar chart of most similar song
most_similar_song = sorted_similar_songs[1]
most_similar_song_name = get_song_name(most_similar_song)
st.write('The most similar song is "', most_similar_song_name, '" with characteristics that look like:')
values2 = get_song_data(most_similar_song)
radar_df2 = pd.DataFrame(dict(
    r=values2,
    theta=spoke_labels
))
fig2 = px.line_polar(radar_df2, r='r', theta='theta', line_close=True, range_r=[0,1])
st.plotly_chart(fig2)


def get_top_similar_song_names():
    data = []
    for i in range(10):
        data.append(get_song_name(sorted_similar_songs[i + 1]))
    return data


top_similar_songs = get_top_similar_song_names()

st.write('The most similar songs are: ', top_similar_songs)

st.write('The following chart shows the similarity percentage of each song:')
fig3 = go.Figure([go.Bar(x=top_similar_songs, y=[sorted_similar_songs[x][1] for x in range(1,11)])])
st.plotly_chart(fig3)

