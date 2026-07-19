import pandas as pd
import streamlit as st

@st.cache_data
def _load_real_data():
    df1 = pd.read_parquet("data/data_part_0.parquet")
    df2 = pd.read_parquet("data/data_part_1.parquet")
    return pd.concat([df1, df2], ignore_index=True)

# Keep a reference to the original read_csv function
_original_read_csv = pd.read_csv

def custom_read_csv(path, *args, **kwargs):
    if path == "data/cleaned_data.csv":
        return _load_real_data()
    return _original_read_csv(path, *args, **kwargs)

# Override the pandas read_csv function with our custom function
pd.read_csv = custom_read_csv