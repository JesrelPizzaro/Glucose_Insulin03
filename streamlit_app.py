import streamlit as st
import pandas as pd
import math
from pathlib import Path
from pandas import read_csv

# Set the title and favicon that appear in the browser's tab bar
st.set_page_config(
    page_title='Glucose and Insulin Simulation',
    page_icon=':chart_with_upwards_trend:'
)

# Title of the app
st.title('Glucose and Insulin Simulation')

# Load the data file
@st.cache_data
def load_data():
    file_path = 'glucose_insulin.csv'
    if not Path(file_path).exists():
        st.error("Data file 'glucose_insulin.csv' not found. Please upload it.")
        uploaded_file = st.file_uploader("Upload your glucose_insulin.csv file", type=["csv"])
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file, index_col='time')
        else:
            return None
    return read_csv(file_path, index_col='time')

data = load_data()
if data is None:
    st.stop()

st.write("### Data Preview")
st.dataframe(data.head())

# Parameters for the model
G0 = st.sidebar.slider("Initial Glucose (G0)", min_value=200, max_value=300, value=270)
k1 = st.sidebar.slider("Rate constant k1", min_value=0.01, max_value=0.05, value=0.02, step=0.01)
k2 = st.sidebar.slider("Rate constant k2", min_value=0.01, max_value=0.05, value=0.02, step=0.01)
k3 = st.sidebar.slider("Rate constant k3", min_value=1e-5, max_value=5e-5, value=1.5e-5, step=1e-5)

params = G0, k1, k2, k3

# Define system creation function
def make_system(params, data):
    G0, k1, k2, k3 = params

    t_0 = data.index[0]
    t_end = data.index[-1]

    Gb = data.glucose[t_0]
    Ib = data.insulin[t_0]
    I = interpolate(data.insulin)

    init = State(G=G0, X=0)

    return System(init=init, params=params,
                  Gb=Gb, Ib=Ib, I=I,
                  t_0=t_0, t_end=t_end, dt=2)

# Define the update function
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

# Simulation runner
def run_simulation(system, update_func):
    t_array = linrange(system.t_0, system.t_end, system.dt)
    frame = TimeFrame(index=t_array, columns=system.init.index)
    frame.iloc[0] = system.init

    for t in t_array[:-1]:
        state = frame.loc[t]
        frame.loc[t + system.dt] = update_func(t, state, system)

    return frame

# Run the simulation
system = make_system(params, data)
results = run_simulation(system, update_func)

# Plot the results
st.write("### Simulation Results")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Glucose plot
ax[0].plot(data.index, data.glucose, 'o', alpha=0.5, label='Glucose Data')
ax[0].plot(results.index, results.G, '-', label='Simulation')
ax[0].set_xlabel('Time (min)')
ax[0].set_ylabel('Glucose (mg/dL)')
ax[0].legend()

# Remote insulin plot
ax[1].plot(results.index, results.X, '-', color='orange', label='Remote Insulin')
ax[1].set_xlabel('Time (min)')
ax[1].set_ylabel('Insulin (arbitrary units)')
ax[1].legend()

st.pyplot(fig)

# Compare with solve_ivp
def slope_func(t, state, system):
    G, X = state
    G0, k1, k2, k3 = system.params 
    I, Ib, Gb = system.I, system.Ib, system.Gb

    dGdt = -k1 * (G - Gb) - X * G
    dXdt = k3 * (I(t) - Ib) - k2 * X

    return dGdt, dXdt

results2, details = run_solve_ivp(system, slope_func, t_eval=results.index)

if details.success:
    st.write("### Comparison with solve_ivp")
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Glucose comparison
    ax[0].plot(results.index, results.G, '--', label='Custom Simulation')
    ax[0].plot(results2.index, results2.G, '-', label='solve_ivp')
    ax[0].set_xlabel('Time (min)')
    ax[0].set_ylabel('Glucose (mg/dL)')
    ax[0].legend()

    # Insulin comparison
    ax[1].plot(results.index, results.X, '--', label='Custom Simulation')
    ax[1].plot(results2.index, results2.X, '-', label='solve_ivp')
    ax[1].set_xlabel('Time (min)')
    ax[1].set_ylabel('Insulin (arbitrary units)')
    ax[1].legend()

    st.pyplot(fig)
else:
    st.error("solve_ivp failed: " + details.message)


