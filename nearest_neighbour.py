import pandas as pd
import streamlit as st
import requests
import numpy as np
import os
import joblib

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
    """Loads the compressed dataset and model from GitHub."""
    # GitHub raw URLs for the files
    dataset_url = "https://raw.githubusercontent.com/<your_github_username>/<repo_name>/main/data_with_attributes_reduced.pkl.gz"
    embeddings_url = "https://raw.githubusercontent.com/<your_github_username>/<repo_name>/main/horse_embeddings.npy"
    model_url = "https://raw.githubusercontent.com/<your_github_username>/<repo_name>/main/nn_model.pkl"

    # File paths
    dataset_file = "data_with_attributes_reduced.pkl.gz"
    embeddings_file = "horse_embeddings.npy"
    model_file = "nn_model.pkl"

    # Download the files if they don't exist
    if not os.path.exists(dataset_file):
        st.write(f"Downloading {dataset_file} from GitHub...")
        download_file_from_github(dataset_url, dataset_file)
    if not os.path.exists(embeddings_file):
        st.write(f"Downloading {embeddings_file} from GitHub...")
        download_file_from_github(embeddings_url, embeddings_file)
    if not os.path.exists(model_file):
        st.write(f"Downloading {model_file} from GitHub...")
        download_file_from_github(model_url, model_file)

    # Load the data and model
    try:
        data = pd.read_pickle(dataset_file, compression="gzip")
        horse_embeddings = np.load(embeddings_file, allow_pickle=False)
        nn_model = joblib.load(model_file)
        return data, horse_embeddings, nn_model
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# ---------------------------------------------------------
# Streamlit App UI
# ---------------------------------------------------------
st.title("Next Nearest Horse Neigh-bour")

# Load data and model (cached)
data, horse_embeddings, nn_model = load_data()

# Check if data loading was successful
if data is None or horse_embeddings is None or nn_model is None:
    st.stop()

# User input for horse name
selected_horse_name = st.text_input("Enter your horse name:")

if selected_horse_name:
    # Check if the horse exists in the dataset
    if selected_horse_name in data['horse_name'].values:
        # Display the searched-for horse's details
        searched_horse = data[data['horse_name'] == selected_horse_name].iloc[0]
        st.write(f"### Searched Horse")
        st.write(f"**Name**: {searched_horse['horse_name']}")

        # Match the embedding using horse_id
        horse_id = searched_horse['horse_id']
        try:
            horse_index = data.reset_index().query(f"horse_id == '{horse_id}'").index[0]
            horse_vector = horse_embeddings[horse_index].reshape(1, -1)
        except IndexError:
            st.error(f"No embedding found for horse '{selected_horse_name}'.")
            st.stop()

        # Get the top 5 nearest neighbors
        distances, indices = nn_model.kneighbors(horse_vector, n_neighbors=5)

        # Display the top 5 closest matches
        st.write(f"### Top 5 Closest Matches")
        for rank, (neighbor_idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
            match = data.iloc[neighbor_idx]
            url = f"https://photofinish.live/horses/{match['horse_id']}"
            st.markdown(
                f"**{rank}. {match['horse_name']}**\n"
                f"- [View Horse Profile]({url})\n"
                f"- **Similarity Score**: {dist:.3f}"
            )
    else:
        st.error(f"Horse name '{selected_horse_name}' not found in the dataset.")
