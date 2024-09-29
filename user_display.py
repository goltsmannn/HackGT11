import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from streamlit_webrtc import webrtc_streamer
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
from record_squats import record_squat_set

st.title("Squat Analyzer")
col1, col2, col3 = st.columns([10, 10, 10])
if 'stream_started' not in st.session_state:
    st.session_state['stream_started'] = False

if 'recording_type' not in st.session_state:
    st.session_state['recording_type'] = 'reference'

async def process(image):
    return await record_squat_set(f'{st.session_state["recording.type"]}.csv', image, num_reps=3)

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")



with col1:
    if st.button("Use prexisting data"):
        st.write("Prexisting data chosen")
with col2:
    if st.session_state['stream_started']:
        stop_stream = st.button("Hide Video Capturing Instruments")
        if stop_stream:
            st.session_state['stream_started'] = False
    else:
        start_stream = st.button("Show Video Capturing Instruments")
        if start_stream:
            st.session_state['stream_started'] = True
with col3:
    if st.session_state['recording_type'] == 'reference':
        record_weights = st.button("Click here to record weighted set (currently awaiting a reference set")
        if record_weights:
            st.session_state['recording_type'] = "weighted"
    else:
        record_reference = st.button("Click here to record reference set (currently awaiting a weighted set")
        if record_reference:
            st.session_state['recording_type'] = "reference"


if st.session_state['stream_started']:
    webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )

st.button("View your results!")

df = pd.DataFrame(
    np.random.randint(50, 100, size=(10, 4)), columns=("Set %d" % i for i in range(4))
)

st.table(df)

sets = df.columns.tolist()
selected_set = st.selectbox("Select a set to view", sets)
unique_sets = df[selected_set].unique()

filtered_df = pd.DataFrame({
    'Score': df[selected_set]
})

st.write(filtered_df)

if st.button("Generate Score Graph for " + selected_set):
    chart = alt.Chart(filtered_df).mark_line().encode(
        x = alt.X('Rep', title = 'Reps', scale=alt.Scale(domain=[0, len(df) - 1])),
        y = alt.Y('Score', title = 'Score', scale=alt.Scale(domain=[0,100]))
    ).properties(
        width=700,
        height=400,
        title=f"Score per Rep for {selected_set}"
    )

    st.altair_chart(chart)