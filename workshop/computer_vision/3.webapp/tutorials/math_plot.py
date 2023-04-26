import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Define the function to be plotted
def func(x, a, b):
    return a * np.sin(b * x)

# Define the range of x values to be plotted
x = np.linspace(-10, 10, 200)

# Set up the Streamlit app
st.title('Interactive Math Function Plot')
a = st.slider('Select a value for "a"', min_value=1, max_value=10, value=5)
b = st.slider('Select a value for "b"', min_value=1, max_value=10, value=5)

# Plot the function with the chosen values of a and b
y = func(x, a, b)
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'y = {a} * sin({b} * x)')
st.pyplot(fig)
