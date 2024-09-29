import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
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