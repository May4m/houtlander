import streamlit

url = "http://35.193.88.23:8000/"


streamlit.markdown(
    f'<meta http-equiv="refresh" content="0; url={url}">',
    unsafe_allow_html=True
)
