import streamlit as st

# Set the page title
st.set_page_config(page_title="Streamlit Example")

# Define some options for a selectbox
options = ["Option 1", "Option 2", "Option 3"]

# Define a slider with some default values
value = st.slider("Select a value", 0, 10, 5)

# Define a checkbox with a default value
checked = st.checkbox("Check me Huray!", True)

# Define a selectbox with the options defined above
selected_option = st.selectbox("Select an option", options)

# Define a button that displays a message when clicked
if st.button("Click me!"):
    st.write(f"You selected {selected_option}, with a value of {value} and checkbox is {checked}")