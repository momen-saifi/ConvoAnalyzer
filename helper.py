from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from textblob import TextBlob
from datetime import timedelta
import plotly.graph_objects as go
import re


extract = URLExtract()

# Function to clean and preprocess messages
def clean_messages(messages, custom_stop_words=None):
    if custom_stop_words is None:
        custom_stop_words = []

    # Additional words and patterns to remove
    additional_stop_words = ['<Media omitted>', 'deleted', 'message']
    numbers_pattern = r'\b\d+\b'  # Regex pattern to match numbers

    cleaned_messages = []
    for message in messages:
        # Remove specific words and numbers
        for word in additional_stop_words + custom_stop_words:
            message = message.replace(word, '')
        # Remove numbers using regex pattern
        message = re.sub(numbers_pattern, '', message)
        # Extract URLs and remove them
        urls = extract.find_urls(message)
        for url in urls:
            message = message.replace(url, '')
        cleaned_messages.append(message)
    return cleaned_messages


# Function to perform topic modeling
def perform_topic_modeling(selected_user, df, num_topics=5):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Clean and preprocess messages
    cleaned_messages = clean_messages(df['message'].values)

    # Vectorize the messages using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=2)
    X = vectorizer.fit_transform(cleaned_messages)

    # Apply LDA to extract topics
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)

    # Get the topics and associated words
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-10 - 1:-1]  # Get top 10 words for each topic
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append(', '.join(top_words))

    return topics


def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]
    words = []
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]
    links = []

    sentiments = []  # List to store message sentiments

    for message in df['message']:
        words.extend(message.split())
        links.extend(extract.find_urls(message))

        # Perform sentiment analysis on each message
        blob = TextBlob(message)
        sentiments.append(blob.sentiment.polarity)

    # Calculate average sentiment
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

    return num_messages, len(words), num_media_messages, len(links), avg_sentiment

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

def sentiment_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    cleaned_messages = clean_messages(df['message'].values)

    # Using TextBlob for sentiment analysis as a fallback
    sentiments = []
    for message in cleaned_messages:
        blob = TextBlob(message)
        sentiments.append(blob.sentiment.polarity)

    df['sentiments'] = sentiments
    return df



def create_gauge_chart(value, max_value, title, color='rgb(31, 119, 180)'):
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={'axis': {'range': [None, max_value]},
               'bar': {'color': color},
               'steps': [
                   {'range': [0, max_value/3], 'color': 'red'},
                   {'range': [max_value/3, 2*max_value/3], 'color': 'yellow'},
                   {'range': [2*max_value/3, max_value], 'color': 'green'}],
               'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': value}
               }
    ))

    return fig


def calculate_user_engagement_metrics(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df['previous_message_time'] = df['date'].shift()
    df['response_time'] = (df['date'] - df['previous_message_time']).fillna(timedelta(seconds=0))

    avg_message_length = df['message'].apply(len).mean()
    avg_response_time = df['response_time'].mean().total_seconds()

    response_time_distribution = df['response_time'].dt.total_seconds()

    return avg_response_time, avg_message_length, response_time_distribution

def create_wordcloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        # Extract emojis from the message using EMOJI_DATA dictionary
        extracted_emojis = [c for c in message if c in emoji.EMOJI_DATA]
        # Store the extracted emojis individually
        emojis.extend(extracted_emojis)

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap

