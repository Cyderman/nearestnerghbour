import pandas as pd
import streamlit as st
import requests
import os

def download_file_from_github(url, destination):
    """Download a file from GitHub and save it locally."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)
    else:
        raise ValueError(f"Failed to download file from GitHub. HTTP Status Code: {response.status_code}")

@st.cache_data
def load_data():
    """Loads the compressed dataset from GitHub."""
    # GitHub raw URL for the file
    url = "https://raw.githubusercontent.com/<your_github_username>/<repo_name>/main/data_with_attributes_reduced.pkl.gz"
    compressed_file = "data_with_attributes_reduced.pkl.gz"

    # Download the file if it doesn't exist locally
    if not os.path.exists(compressed_file):
        st.write(f"Downloading {compressed_file} from GitHub...")
        try:
            download_file_from_github(url, compressed_file)
        except ValueError as e:
            st.error(f"Error during download: {e}")
            return None, None, None, None

    # Load the compressed pickle file
    try:
        data = pd.read_pickle(compressed_file, compression="gzip")
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ---------------------------------------------------------
# Streamlit App UI
# ---------------------------------------------------------
st.title("Horse Similarity Finder")

# Load data (cached)
data = load_data()

# Check if data loading was successful
if data is not None:
    st.write("Dataset loaded successfully!")
    st.write(data.head())
else:
    st.error("Failed to load dataset.")

# Input field for horse name
selected_horse_name = st.text_input("Enter the horse name:")

if selected_horse_name:
    if selected_horse_name in data['horse_name'].values:
        horse_data = data[data['horse_name'] == selected_horse_name]
        st.write(f"Details for '{selected_horse_name}':")
        st.write(horse_data)
    else:
        st.write(f"Horse name '{selected_horse_name}' not found in the dataset.")
