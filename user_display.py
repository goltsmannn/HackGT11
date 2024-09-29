import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("Squat Analyzer")

col1, col2, col3 = st.columns([10, 10, 10])
with col1:
    if st.button("Use prexisting data"):
        st.write("Prexisting data chosen")
with col2:
    if st.button("Record your reference"):
        st.write("Ready to record a reference")

st.button ("Begin Recording")
webrtc_streamer(key="streamer", sendback_audio=False)

st.button("View your results!")