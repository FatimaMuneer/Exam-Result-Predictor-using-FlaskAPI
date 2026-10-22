import streamlit as st

# BAD — counter resets to 0 every time the button is clicked
counter = 0
if st.button("Click me"):
    counter += 1
st.write(counter)  # always shows 1

# GOOD — counter survives re-runs
if "counter" not in st.session_state:
    st.session_state.counter = 0  # only set once

if st.button("Click me"):
    st.session_state.counter += 1
st.write(st.session_state.counter)  # correctly increments