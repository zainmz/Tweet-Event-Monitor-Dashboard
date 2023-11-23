# Tweet Event Monitoring Dashboard

## Overview
This repository contains the code for a Streamlit-based dashboard designed to monitor and analyze tweets related to the Sri Lankan political and economic crisis of 2022. The dashboard provides insights into social media discussions, displaying data visualizations, sentiment analysis, geographic distributions, and predictive models.

## Features
- **Data Overview:** An overview of the dataset including total tweets and coverage dates.
- **Hashtag Analysis:** Visualization of trending hashtags in the form of a word cloud.
- **Geographic Mapping:** Displays the geographic distribution of tweets using Folium maps.
- **Tweet Activity Timeline:** A line chart showing the number of tweets over time.
- **Sentiment Analysis:** Classification of tweets into positive, negative, and neutral sentiments.
- **Geo Sentiment Distribution:** Map visualization showing the geographic distribution of sentiments.
- **Tweet Prediction:** Forecasting the number of tweets for the next hour using the ARIMA model.
- **Raw Tweet Data Display:** Shows the raw tweet data in a table format.

## Data Collection
The data was manually downloaded from Kaggle due to the unavailability of the free Twitter API. The dataset reflects public sentiment about the ongoing crisis in Sri Lanka and is suitable for sentiment analysis research. The dataset contains 10,000 tweets up to 11/07/2022. 
Source: [Twitter Dataset - Sri Lanka Crisis](https://www.kaggle.com/datasets/vishesh1412/twitter-dataset-sri-lanka-crisis) by Vishesh Thakur.

## Installation and Usage
1. Clone the repository:
```git clone https://github.com/zainmz/Tweet-Event-Monitor-Dashboard```
2. Install the required dependencies:
```pip install -r requirements.txt```
3. Run the Streamlit app:
```streamlit run dashboard.py```

## Technologies Used
- Python
- Streamlit
- Pandas
- Matplotlib
- Plotly
- Folium
- Spacy
- ARIMA (Statsmodels)

## Acknowledgments
- This project uses data from [Kaggle](https://www.kaggle.com/datasets/vishesh1412/twitter-dataset-sri-lanka-crisis) provided by Vishesh Thakur.
- Geolocation data sourced from [Simplemaps](https://simplemaps.com/data/world-cities).
- Country data obtained from [PyCountry](https://github.com/flyingcircusio/pycountry).
- Natural Language Processing with [Spacy](https://spacy.io/).

