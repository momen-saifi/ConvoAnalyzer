import streamlit as st
import preprocessor
import helper
import io
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Load the header image
header_image_path = 'header_1.png'
header_image_width = 600
header_image_height = 100

header_image = Image.open(header_image_path)
resized_header_image = header_image.resize((header_image_width, header_image_height))

st.image(resized_header_image)

# Set background image using st.markdown
st.markdown(
    """
    <style>
    body {
        background-image: url('C:\\Users\\Momeen\\PycharmProjects\\whatsapp-chat-analyzer\\b1.jpg');  /* Path to your background image */
        background-size: cover;
        background-color: #ADD8E6;  /* Background color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("ConvoAnalyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    try:
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode("utf-8")
        df = preprocessor.preprocess(data)

        selected_year = st.sidebar.selectbox("Select Year", ["All"] + df['year'].unique().tolist())
        analyze_by_month = st.sidebar.checkbox("Enable Month Selection")

        if analyze_by_month:
            selected_month = st.sidebar.selectbox("Select Month", ["All"] + df['month'].unique().tolist())
        else:
            selected_month = "All"

        if selected_year != "All":
            filtered_df = df[df['year'] == selected_year]
            if selected_month != "All":
                filtered_df = filtered_df[filtered_df['month'] == selected_month]
        else:
            filtered_df = df

        user_list = filtered_df['user'].unique().tolist()
        user_list.remove('group_notification')
        user_list.sort()
        user_list.insert(0, "Overall")

        selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

        if st.sidebar.button("Show Analysis"):
            try:
                num_messages, words, num_media_messages, num_links, avg_sentiment = helper.fetch_stats(selected_user,
                                                                                                          filtered_df)

                st.title("Top Statistics")
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("Total Messages", num_messages)
                with col2:
                    st.metric("Total Words", words)
                with col3:
                    st.metric("Media Shared", num_media_messages)
                with col4:
                    st.metric("Links Shared", num_links)
                with col5:
                    st.metric("Average Sentiment", round(avg_sentiment, 2))

                # Monthly timeline
                st.title("Monthly Timeline")
                timeline = helper.monthly_timeline(selected_user, filtered_df)
                fig, ax = plt.subplots()
                ax.plot(timeline['time'], timeline['message'], color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

                # Daily timeline
                st.title("Daily Timeline")
                daily_timeline = helper.daily_timeline(selected_user, filtered_df)
                fig, ax = plt.subplots()
                ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

                # Topic Modeling
                st.title("Topic Modeling")
                num_topics = st.slider("Select Number of Topics", min_value=1, max_value=10, value=5)
                try:
                    topics = helper.perform_topic_modeling(selected_user, filtered_df, num_topics)

                    if topics:
                        st.write("Topics:")
                        for i, topic in enumerate(topics):
                            st.write(f"Topic {i + 1}: {topic}")
                    else:
                        st.info("No topics to show.")
                except ValueError as e:
                    st.error("An error occurred during topic modeling: {}".format(str(e)))
                    st.info("Try adjusting min_df or max_df parameters for vectorization.")

                # Sentiment Analysis
                df = helper.sentiment_analysis(selected_user, filtered_df)
                st.title("Sentiment Analysis")

                # Sentiment Emoji Mapping
                sentiment_emojis = {
                    'positive': '😄',
                    'neutral': '😐',
                    'negative': '😢'
                }

                # Map sentiment labels to emojis
                df['sentiment_emoji'] = df['sentiments'].apply(lambda x: sentiment_emojis['positive'] if x > 0
                                                               else sentiment_emojis['neutral'] if x == 0
                                                               else sentiment_emojis['negative'])

                # Sentiment Distribution Pie Chart
                fig_sentiment = px.pie(df, names='sentiment_emoji', title='Pie Chart',
                                       hole=0.3, color_discrete_map=sentiment_emojis)
                st.plotly_chart(fig_sentiment)

                # Sentiment Bar Chart
                fig_sentiment_bar = px.histogram(df, x='sentiments', nbins=30, title='Bar Analysis')
                st.plotly_chart(fig_sentiment_bar)

                # User Engagement Metrics
                st.title("User Engagement Metrics")
                avg_response_time, avg_message_length, response_time_distribution = helper.calculate_user_engagement_metrics(
                    selected_user, filtered_df)

                # Normalize average response time to fit within [0.0, 1.0] range
                max_response_time = 24 * 60 * 60  # Maximum response time in seconds (24 hours)
                normalized_response_time = avg_response_time / max_response_time

                # Display average response time as a gauge chart
                st.subheader("Average Response Time")
                st.plotly_chart(helper.create_gauge_chart(normalized_response_time, 1, "Average Response Time"))

                # Display average message length
                st.subheader("Average Message Length")
                st.write(f"Average Message Length: {avg_message_length:.2f} characters")

                # Response Time Distribution
                st.subheader("Response Time Distribution")
                fig_response_time = px.histogram(response_time_distribution, nbins=30,
                                                 labels={'value': 'Response Time (seconds)'},
                                                 color_discrete_sequence=['#FFBB33'])
                fig_response_time.update_xaxes(title_text='Response Time (seconds)')
                fig_response_time.update_yaxes(title_text='Count')
                st.plotly_chart(fig_response_time)

                # Activity map
                st.title('Activity Map')
                col1, col2 = st.columns(2)

                with col1:
                    st.header("Most busy day")
                    busy_day = helper.week_activity_map(selected_user, filtered_df)
                    fig, ax = plt.subplots()
                    ax.bar(busy_day.index, busy_day.values, color='purple')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)

                with col2:
                    st.header("Most busy month")
                    busy_month = helper.month_activity_map(selected_user, filtered_df)
                    fig, ax = plt.subplots()
                    ax.bar(busy_month.index, busy_month.values, color='orange')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)

                st.title("Weekly Activity Map")
                user_heatmap = helper.activity_heatmap(selected_user, filtered_df)
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)

                # Finding the busiest users in the group (Group level)
                if selected_user == 'Overall':
                    st.title('Most Busy Users')
                    x, new_df = helper.most_busy_users(filtered_df)
                    fig, ax = plt.subplots()

                    col1, col2 = st.columns(2)

                    with col1:
                        ax.bar(x.index, x.values, color='red')
                        plt.xticks(rotation='vertical')
                        st.pyplot(fig)
                    with col2:
                        st.dataframe(new_df)

                # WordCloud
                st.title("Wordcloud")
                wordcloud = helper.create_wordcloud(selected_user, filtered_df)
                st.image(wordcloud.to_image())

                # Most common words
                most_common_df = helper.most_common_words(selected_user, filtered_df)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1])
                plt.xticks(rotation='vertical')
                st.title('Most Common Words')
                st.pyplot(fig)

                try:
                    # Emoji analysis
                    emoji_df = helper.emoji_helper(selected_user, filtered_df)
                    st.title("Emoji Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(emoji_df)
                    with col2:
                        fig, ax = plt.subplots()
                        ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f", textprops={'fontsize': 14})
                        st.pyplot(fig)
                except Exception as e:
                    st.error("An error occurred during emoji analysis: {}".format(str(e)))

                # Export button for analysis results
                csv_file = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Analysis Results", csv_file, "analysis_results.csv", "text/csv")

                # Visualization download links
                st.markdown("### Download Visualizations")

                # Download monthly timeline chart
                monthly_timeline_fig = px.line(timeline, x='time', y='message', title='Monthly Timeline')
                monthly_timeline_fig_bytes = io.BytesIO()
                monthly_timeline_fig.write_image(monthly_timeline_fig_bytes, format='png')
                st.download_button("Download Monthly Timeline Chart", monthly_timeline_fig_bytes.getvalue(),
                                   "monthly_timeline_chart.png", "image/png")

                # Download daily timeline chart
                daily_timeline_fig = px.line(daily_timeline, x='only_date', y='message', title='Daily Timeline')
                daily_timeline_fig_bytes = io.BytesIO()
                daily_timeline_fig.write_image(daily_timeline_fig_bytes, format='png')
                st.download_button("Download Daily Timeline Chart", daily_timeline_fig_bytes.getvalue(),
                                   "daily_timeline_chart.png", "image/png")

                # Download sentiment analysis chart
                sentiment_fig_bytes = io.BytesIO()
                fig_sentiment.write_image(sentiment_fig_bytes, format='png')
                st.download_button("Download Sentiment Analysis Chart", sentiment_fig_bytes.getvalue(),
                                   "sentiment_analysis_chart.png", "image/png")

                # Add download buttons for other visualizations as needed

            except Exception as e:
                st.error("An error occurred: {}".format(str(e)))

    except Exception as e:
        st.error("An error occurred: {}".format(str(e)))
