import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Title and description
st.title("Monte Carlo Simulation for Risk Analysis")
st.write("""This app uses Monte Carlo Simulation to analyze risk by modeling possible outcomes based on input assumptions.

### How it works:
1. Define the distribution of the uncertain variables.
2. Simulate multiple outcomes.
3. Analyze the results.
""")

# User inputs
st.sidebar.header("Simulation Parameters")

num_simulations = st.sidebar.number_input("Number of Simulations", min_value=100, max_value=100000, value=1000, step=100)
mean = st.sidebar.number_input("Mean of Distribution", value=100.0)
std_dev = st.sidebar.number_input("Standard Deviation of Distribution", value=20.0)

# Running the simulation
st.header("Simulating Outcomes")
st.write("Running the Monte Carlo simulation with the following parameters:")
st.write(f"Number of simulations: {num_simulations}")
st.write(f"Mean: {mean}, Standard Deviation: {std_dev}")

# Perform simulation
np.random.seed(42)  # For reproducibility
simulated_data = np.random.normal(loc=mean, scale=std_dev, size=num_simulations)

# Display results
st.header("Simulation Results")

# Summary statistics
st.subheader("Summary Statistics")
results_summary = {
    "Mean": np.mean(simulated_data),
    "Median": np.median(simulated_data),
    "Standard Deviation": np.std(simulated_data),
    "5th Percentile": np.percentile(simulated_data, 5),
    "95th Percentile": np.percentile(simulated_data, 95)
}
st.write(pd.DataFrame.from_dict(results_summary, orient='index', columns=['Value']))

# Histogram
st.subheader("Histogram of Results")
fig, ax = plt.subplots()
ax.hist(simulated_data, bins=50, color='blue', edgecolor='black', alpha=0.7)
ax.set_title("Simulation Results Distribution")
ax.set_xlabel("Outcome")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Download option
st.subheader("Download Simulated Data")
dataframe = pd.DataFrame(simulated_data, columns=["Simulated Outcome"])
csv = dataframe.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="simulated_data.csv",
    mime="text/csv"
)
