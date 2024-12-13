import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modsim import *
import numpy as np

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Glucose and Insulin Simulation",
    page_icon=":chart_with_upwards_trend:"
)

st.title("Glucose and Insulin Model Simulation")

# Load data
try:
    data = pd.read_csv('glucose_insulin.csv', index_col='time')
    st.write("Loaded data:")
    st.dataframe(data.head())
except FileNotFoundError:
    st.error("The file 'glucose_insulin.csv' was not found. Please upload it.")
    st.stop()

# Model parameters
G0 = 270
k1 = 0.02
k2 = 0.02
k3 = 1.5e-5
params = G0, k1, k2, k3

def make_system(params, data):
    G0, k1, k2, k3 = params
    t_0 = data.index[0]
    t_end = data.index[-1]
    Gb = data.glucose.iloc[0]
    Ib = data.insulin.iloc[0]
    I = interpolate(data.insulin)
    init = State(G=G0, X=0)
    return System(init=init, params=params, Gb=Gb, Ib=Ib, I=I, t_0=t_0, t_end=t_end, dt=2)

def update_func(t, state, system):
    G, X = state
    G0, k1, k2, k3 = system.params
    I, Ib, Gb = system.I, system.Ib, system.Gb
    dt = system.dt
    dGdt = -k1 * (G - Gb) - X * G
    dXdt = k3 * (I(t) - Ib) - k2 * X
    G += dGdt * dt
    X += dXdt * dt
    return State(G=G, X=X)

def run_simulation(system, update_func):
    t_array = linrange(system.t_0, system.t_end, system.dt)
    frame = TimeFrame(index=t_array, columns=system.init.index)
    frame.iloc[0] = system.init
    for i, t in enumerate(t_array[:-1]):
        state = frame.iloc[i]
        frame.iloc[i+1] = update_func(t, state, system)
    return frame

# Create system and run simulation
system = make_system(params, data)
results = run_simulation(system, update_func)

# Plot results
fig, ax = plt.subplots(2, 1, figsize=(10, 6))

# Glucose Data and Simulation
data.glucose.plot(style='o', alpha=0.5, label='Glucose Data', ax=ax[0])
results.G.plot(style='-', color='C0', label='Simulation', ax=ax[0])
ax[0].set_title("Glucose Levels")
ax[0].set_xlabel("Time (min)")
ax[0].set_ylabel("Concentration (mg/dL)")
ax[0].legend()

# Remote Insulin Simulation
results.X.plot(color='C1', label='Remote Insulin', ax=ax[1])
ax[1].set_title("Remote Insulin Levels")
ax[1].set_xlabel("Time (min)")
ax[1].set_ylabel("Concentration (arbitrary units)")
ax[1].legend()

# Display the plot
st.pyplot(fig)

# Summary
st.subheader("Simulation Summary")
st.write("The model simulation predicts glucose and insulin dynamics based on the provided parameters and input data.")
