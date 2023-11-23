import re
import spacy
import pandas as pd
import pycountry
from fuzzywuzzy import process


########################################################################################################################
#                                       MAIN PROCESS
########################################################################################################################

def main():
    ####################################################################################################################
    #                                       FUNCTIONS
    ####################################################################################################################
    # Function to extract place names from a text using NER
    def extract_place_names(text):
        doc = nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ == "GPE"]

    # Function to check if a place name is a @mention
    def is_mention(place_name):
        return re.match(r'@\w+', place_name) is not None

    # Function to check if a place name contains a URL
    def contains_url(place_name):
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.search(url_pattern, place_name) is not None

    # Function to remove emojis
    def remove_emojis(text):
        return emoji_pattern.sub(r'', text)

    # Function to find the closest country name
    def find_closest_country(place_name):
        closest_country, _ = process.extractOne(place_name, country_names)
        return closest_country

    def fill_geo_coords(unique_place_names_df):
        world_cities_df = pd.read_csv("worldcities.csv")

        # Create a combined city-country column for easier matching
        world_cities_df['city_country'] = world_cities_df['city'] + ', ' + world_cities_df['country']
        world_cities_df['country_only'] = world_cities_df['country']

        # Function to find coordinates for a given place name
        def find_coordinates(place_name):
            # Try matching with city-country combination
            match = world_cities_df[world_cities_df['city_country'] == place_name]
            if not match.empty:
                return match.iloc[0]['lat'], match.iloc[0]['lng']

            # Try matching with country only
            match = world_cities_df[world_cities_df['country_only'] == place_name]
            if not match.empty:
                return match.iloc[0]['lat'], match.iloc[0]['lng']

            # No match found
            return None, None

        # Apply the function to each place name in 'unique_place_names_df'
        unique_place_names_df['latitude'], unique_place_names_df['longitude'] = zip(
            *unique_place_names_df['unique_place_names'].apply(find_coordinates))

        return unique_place_names_df

    tweets_df = pd.read_csv("SriLankaTweets.csv")

    # Convert 'created_at' from Unix timestamp (milliseconds) to datetime
    tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'], unit='ms')

    # Load the English NLP model
    nlp = spacy.load("en_core_web_sm")

    # Apply the function to tweets
    tweets_df['place_names'] = tweets_df['tweet'].apply(extract_place_names)
    tweets_df = tweets_df[tweets_df['place_names'].str.len() > 0]

    # List of columns to keep
    columns_to_keep = ['tweet', 'created_at', 'place_names', 'hashtags']
    tweets_df = tweets_df[columns_to_keep]

    # Flatten the list of place names
    all_place_names = [place for sublist in tweets_df['place_names'] for place in sublist]

    # Remove duplicates and create a new DataFrame
    unique_place_names_df = pd.DataFrame({'unique_place_names': list(set(all_place_names))})

    # Filter out @mentions
    unique_place_names_df = unique_place_names_df[~unique_place_names_df['unique_place_names'].apply(is_mention)]

    # Filter out entries with URLs
    unique_place_names_df = unique_place_names_df[~unique_place_names_df['unique_place_names'].apply(contains_url)]

    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\u200d"
        "\u2640-\u2642"
        "\U0001F1F2-\U0001F1F4"  # Macau flag
        "\U0001F1E6-\U0001F1FF"  # flags
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\u2600-\u26FF"  # Miscellaneous Symbols
        "\u2700-\u27BF"  # Dingbats
        "\U0001F1E0-\U0001F1FF"  # Flags
        "]+", flags=re.UNICODE)

    # Apply the emoji removal function to each place name
    unique_place_names_df['unique_place_names'] = unique_place_names_df['unique_place_names'].apply(remove_emojis)

    # Create a list of all country names with PyCountry
    country_names = [country.name for country in pycountry.countries]

    # Apply country identification for each place name using pycountry list
    unique_place_names_df['identified_country'] = unique_place_names_df['unique_place_names'].apply(
        find_closest_country)

    unique_place_names_df = fill_geo_coords(unique_place_names_df)

    # for the countries that do not have latitude and longitude, replace their unique_place_names value with the
    # value of identified country Replace unique_place_names with identified_country where coordinates are missing
    mask = unique_place_names_df['latitude'].isna() & unique_place_names_df['longitude'].isna()
    unique_place_names_df.loc[mask, 'unique_place_names'] = unique_place_names_df.loc[mask, 'identified_country']

    unique_place_names_df = fill_geo_coords(unique_place_names_df)

    # Step 1: Create a mapping from place names to coordinates
    place_to_coords = {
        row['unique_place_names']: (row['latitude'], row['longitude'])
        for index, row in unique_place_names_df.iterrows()
    }

    # Step 2: Assign coordinates to each tweet
    def assign_coords(place_names):
        # Find the first place name in the list with available coordinates
        for name in place_names:
            if name in place_to_coords and place_to_coords[name] != (None, None):
                return place_to_coords[name]
        return None, None

    # Apply the function to the 'place_names' column of tweets_df
    coords = tweets_df['place_names'].apply(assign_coords)
    tweets_df['latitude'], tweets_df['longitude'] = zip(*coords)

    return unique_place_names_df, tweets_df
