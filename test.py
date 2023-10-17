import streamlit as st
st.markdown(
    """
    <style>
    body {
        background-color: #ADD8E6;
        margin: 0;
        padding: 0;
    }
    .reportview-container {
        background: url('path_to_your_local_background_image.jpg');  /* Path to your local background image */
        background-size: cover;
    }
    .sidebar .sidebar-content {
        background-color: rgba(255, 255, 255, 0.8);  /* Sidebar background color with opacity */
    }
    </style>
    """,
    unsafe_allow_html=True
)
