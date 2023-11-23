import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import pandas as pd
import itertools
import folium
import ast
import re

from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.metric_cards import style_metric_cards
from statsmodels.tsa.arima.model import ARIMA
from streamlit_folium import st_folium
from locationRetrieval import main
from folium.plugins import HeatMap
from wordcloud import WordCloud
from collections import Counter
from textblob import TextBlob

########################################################################################################################
#                                       MAIN PAGE CONFIGURATION
########################################################################################################################
st.set_page_config(page_title="Social Media Event Monitoring Dashboard",
                   page_icon="📊",
                   layout="wide")


########################################################################################################################
#                                       DATA LOADING
########################################################################################################################
# Function to load data with caching
@st.cache_data
def load_data():
    tweets = pd.read_csv("SriLankaTweets.csv")
    tweets['created_at'] = pd.to_datetime(tweets['created_at'], unit='ms')
    tweets['date'] = pd.to_datetime(tweets['date'])
    location_data, tweets_with_places = main()
    return location_data, tweets_with_places, tweets


with st.spinner("Refreshing data...this will take a few minutes"):
    # Load data
    location_data, tweets_with_places, tweets = load_data()

# Interactive Filters
st.sidebar.header("Filters")

# Use columns to split the layout
col1, col2 = st.columns(2)

########################################################################################################################
#                                       DATA OVERVIEW
########################################################################################################################
with col1:
    st.subheader("Sri Lankan Political and Economic Crisis", divider='green')
    st.markdown("This dashboard provides insights into the twitter discussions related to the Sri Lankan"
                " political and economic crisis of 2022")

    # Function to safely convert string representation of lists into actual lists
    def safe_literal_eval(s):
        try:
            return ast.literal_eval(s)
        except:
            return []

    # Extract and count hashtags
    all_hashtags = [hashtag for hashtags_list in tweets['hashtags'].apply(safe_literal_eval) for hashtag in
                    hashtags_list]
    hashtag_counts = Counter(all_hashtags)

    # Display as a word cloud
    st.subheader("Hashtag Word Cloud", divider='green')
    wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate_from_frequencies(
        hashtag_counts)
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()
    st.pyplot(plt)

# Column 2: Data Overview
with col2:
    st.subheader("Data Overview", divider='green')
    # Total number of tweets

    total_tweets = len(tweets)
    st.metric(label="Number of Tweets", value=total_tweets)

    start_date = tweets['created_at'].min().strftime('%Y-%m-%d')

    date_range = f"{start_date}"
    st.metric(label="Coverage Date", value=date_range)

    style_metric_cards(background_color="#000000",
                       border_left_color="#049204",
                       border_color="#0E0E0E"
                       )

########################################################################################################################
#                                       DATA GEO MAPPING
########################################################################################################################


# Extracting the first place name and corresponding coordinates
tweets_with_places['first_place_name'] = tweets_with_places['place_names'].apply(lambda x: x[0] if x else None)
tweets_with_places = tweets_with_places.dropna(subset=['latitude', 'longitude'])

# Extract unique locations
unique_locations = tweets_with_places[['first_place_name', 'latitude', 'longitude']].drop_duplicates()

# Drop any rows with missing coordinates
unique_locations = unique_locations.dropna(subset=['latitude', 'longitude'])

# Initialize a map object
m = folium.Map(location=[0, 0], zoom_start=2)
# Iterate and add markers for unique locations
for idx, row in unique_locations.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

# Display the map in Streamlit
st.header("Geographic Distribution of Tweets", divider="green")

# Tweet Heatmap
# Create a list of coordinates for the heatmap
heatmap_data = [[row['latitude'],
                 row['longitude']] for idx, row in tweets_with_places.dropna(subset=['latitude',
                                                                                     'longitude']).iterrows()]
# Add heatmap layer
HeatMap(heatmap_data).add_to(m)

# Layer control
folium.LayerControl().add_to(m)
st_folium(m, height=500, use_container_width=True)

########################################################################################################################
#                                       TWEET TIME PLOT
########################################################################################################################
st.header("Tweet Activity Over Time", divider='green')
# Group data by hour for the plot
tweets['hour'] = tweets['created_at'].dt.hour
tweets_per_hour = tweets.groupby('hour').size().reset_index(name='count')

# Create a time series plot
fig = px.line(tweets_per_hour, x='hour', y='count', title='Number of Tweets Over Time',
              labels={'hour': 'Hour of the Day', 'count': 'Number of Tweets'})

# Update the layout of the plot
fig.update_layout(
    xaxis_title="Hour of the Day",
    yaxis_title="Number of Tweets",
    xaxis=dict(
        tickmode='linear',
        ticks='outside',
        tick0=0,
        dtick=1
    )
)

# Display the plot in Streamlit
st.plotly_chart(fig, use_container_width=True)


########################################################################################################################
#                                       TWEET SENTIMENT ANALYSIS
########################################################################################################################
# Function to classify sentiment
def classify_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'


# Use columns to split the layout
col1, col2 = st.columns(2)

min_time = tweets['created_at'].dt.time.min()
max_time = tweets['created_at'].dt.time.max()

# Sidebar inputs for start and end times
start_time = st.sidebar.time_input("Coverage Start time", value=min_time)
end_time = st.sidebar.time_input("Coverage End time", value=max_time)

# Apply sentiment analysis to each tweet
tweets['sentiment'] = tweets['tweet'].apply(classify_sentiment)
tweets_with_places['sentiment'] = tweets_with_places['tweet'].apply(classify_sentiment)

# Filter the DataFrame based on the selected times
filtered_tweets = tweets[(tweets['created_at'].dt.time >= start_time) & (tweets['created_at'].dt.time <= end_time)]

with col1:
    st.header("Sentiment Analysis", divider='green')

    if not filtered_tweets.empty:
        # Count sentiments
        sentiment_counts = filtered_tweets['sentiment'].value_counts()

        # Pie chart for sentiment distribution
        fig = px.pie(names=sentiment_counts.index, values=sentiment_counts.values, title='Sentiment Distribution')
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.write("No data available for the selected date range.")

with col2:
    st.subheader("Geo Sentiment Distribution", divider='green')
    m = folium.Map(location=[0, 0], zoom_start=2)
    # Loop through the DataFrame and add markers
    for index, row in tweets_with_places.iterrows():
        # Assign colors based on sentiment
        if row['sentiment'] == 'positive':
            color = 'darkblue'
        elif row['sentiment'] == 'negative':
            color = 'pink'
        else:  # Neutral
            color = 'lightblue'

        # Create a circle marker for each tweet
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)

    # Add layer control to the map (optional)
    folium.LayerControl().add_to(m)

    # Display the map in Streamlit
    st_folium(m, width=700, height=500)

# Create tabs for Positive and Negative tweets
tab_positive, tab_negative = st.tabs(["Positive Tweets", "Negative Tweets"])

# Display Positive Tweets in a scrollable table
with tab_positive:
    positive_tweets = filtered_tweets[filtered_tweets['sentiment'] == 'positive']
    if not positive_tweets.empty:
        # Reset index and drop the index column
        st.dataframe(positive_tweets[['tweet']].reset_index(drop=True), height=300, use_container_width=True)
    else:
        st.write("No positive tweets available for the selected date range.")

# Display Negative Tweets in a scrollable table
with tab_negative:
    negative_tweets = filtered_tweets[filtered_tweets['sentiment'] == 'negative']
    if not negative_tweets.empty:
        # Reset index and drop the index column
        st.dataframe(negative_tweets[['tweet']].reset_index(drop=True), height=300, use_container_width=True)
    else:
        st.write("No negative tweets available for the selected date range.")


def is_mention(place_name):
    return re.match(r'@\w+', place_name) is not None


def extract_country(place_names_list):
    # Filter out @mentions and return the first valid place name
    return next((name for name in place_names_list if not is_mention(name)), None)


tweets_with_places['country'] = tweets_with_places['place_names'].apply(extract_country)

# Extracting the first place name as the country
tweets_with_places['country'] = tweets_with_places['place_names'].apply(extract_country)
# Group by country and sentiment, then count the tweets
sentiment_counts_by_country = tweets_with_places.groupby(['country', 'sentiment']).size().unstack(fill_value=0)
# Sort the DataFrame for better visualization
sentiment_counts_by_country.sort_values(by=['positive', 'negative'], ascending=False, inplace=True)
# Use columns to split the layout
col1, col2 = st.columns(2)

with col1:
    # Sort by 'positive' sentiment and take the top 5
    top_positive_countries = sentiment_counts_by_country.sort_values(by='positive', ascending=False).head(5)

    # Create a bar plot for the top 5 positive sentiment countries
    fig_pos = px.bar(top_positive_countries, y='positive', x=top_positive_countries.index,
                     title="Top 5 Countries with Most Positive Tweets",
                     color_discrete_sequence=['blue'])  # Set color to blue
    st.plotly_chart(fig_pos, use_container_width=True)

with col2:
    # Sort by 'negative' sentiment and take the top 5
    top_negative_countries = sentiment_counts_by_country.sort_values(by='negative', ascending=False).head(5)

    # Create a bar plot for the top 5 negative sentiment countries
    fig_neg = px.bar(top_negative_countries, y='negative', x=top_negative_countries.index,
                     title="Top 5 Countries with Most Negative Tweets",
                     color_discrete_sequence=['pink'])  # Set color to pink
    st.plotly_chart(fig_neg, use_container_width=True)

########################################################################################################################
#                                       TWEET PREDICTION FOR NEXT HOUR
########################################################################################################################

st.subheader("Number of Tweets for the Next Hour (ARIMA Model)", divider='green')
# Convert 'date' to datetime format and set as index
tweets['datetime'] = pd.to_datetime(tweets['date'])
tweets.set_index('datetime', inplace=True)

# Count tweets per hour
tweet_counts = tweets.resample('H').size()

# Create DataFrame for the time series
time_series_df = pd.DataFrame({'tweet_count': tweet_counts})

# Determine the best ARIMA order
p = range(0, 3)
q = range(0, 3)
pq = itertools.product(p, q)
best_aic = float("inf")
best_order = None

for order in pq:
    try:
        model = ARIMA(time_series_df['tweet_count'], order=(order[0], 0, order[1]))
        model_fit = model.fit()
        if model_fit.aic < best_aic:
            best_aic = model_fit.aic
            best_order = order
    except:
        continue

st.write(f'Best ARIMA Order: {best_order}, AIC: {best_aic}')

# Fit the best model
best_model = ARIMA(time_series_df['tweet_count'], order=(best_order[0], 0, best_order[1]))
best_model_fit = best_model.fit()

# Forecast the next hour
forecast = best_model_fit.forecast(steps=1)
st.write(f"Forecast for the next hour: {round(forecast[0])}")

# Plotting
plot_data = time_series_df.copy()
plot_data.loc[time_series_df.index[-1] + pd.Timedelta(hours=1), 'tweet_count'] = forecast[0]
st.line_chart(plot_data['tweet_count'])

########################################################################################################################
#                                               RAW TWEET DATA
########################################################################################################################

# Display the raw tweet data
st.subheader("Raw Tweet Data", divider='green')
raw_tweet_data = tweets_with_places.copy()
raw_tweet_data = raw_tweet_data.drop(['place_names'], axis=1)
filtered_df = dataframe_explorer(raw_tweet_data, case=False)
st.dataframe(raw_tweet_data, use_container_width=True)
