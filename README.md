# Welcome to Streamlit Glucose and Insulin
# Modeling and Simulation in Python

# A simple Streamlit app showing the previous chapter presents the minimal model of the glucose-insulin system and introduces a tool we need to implement it: interpolation.

In this chapter, we'll implement the model two ways:
We'll start by rewriting the differential equations as difference equations; then we'll solve the difference equations using a version of run_simulation similar to what we have used in previous chapters.
Then we'll use a new SciPy function, called solve_ivp, to solve the differential equation using a better algorithm.
We'll see that solve_ivp is faster and more accurate than run_simulation. As a result, we will use it for the models in the rest of the book.

The Update Function
The minimal model is expressed in terms of differential equations:
      dG/dt=−k1[G(t)−Gb]−X(t)G(t) 
      dX/dt=k3[I(t)−Ib]−k2X(t)

To simulate this system, we will rewrite them as difference equations. If we multiply both sides by  dt , we have:
dG=[−k1[G(t)−Gb]−X(t)G(t)]dt 
dX=[k3[I(t)−Ib]−k2X(t)]dt 

If we think of  dt  as a small step in time, these equations tell us how to compute the corresponding changes in  G  and  X . Here's an update function that computes these changes.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gdp-dashboard-template.streamlit.app/)


