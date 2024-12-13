import streamlit as st
import pandas as pd
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Glucose and Insulin',
    page_icon=':Model Simulation :', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

# download modsim.py if necessary

from os.path import basename, exists

def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve
        local, _ = urlretrieve(url, filename)
        print('Downloaded ' + local)

# -------------------------------------------------------------------------------
# import functions from modsim

from modsim import *
from pandas import read_csv

data = read_csv('glucose_insulin.csv', index_col='time');

Implementing the Model
To get started, let's assume that the parameters of the model are known. We'll implement the model and use it to generate time series for G and X. Then we'll see how we can choose parameters that make the simulation fit the data.

Here are the parameters.
G0 = 270
k1 = 0.02
k2 = 0.02
k3 = 1.5e-05
# -------------------------------------------------------------------------------------
#I'll put these values in a sequence which we'll pass to make_system:
params = G0, k1, k2, k3
Here's a version of make_system that takes params and data as parameters.
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
    # -------------------------------------------------------------------------------------
    #




   
