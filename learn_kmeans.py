import numpy as np
import matplotlib.pyplot as plt
import joblib 
from sklearn.manifold import TSNE
from collections import defaultdict
import torch 
from sklearn.cluster import KMeans

import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import copy 

ROOT_DIR = "speakers_hubert_features_personalized"
SPEAKERS = ["p376"] 
N_CLUSTERS = 50 
KMEANS_MODEL_PATH = "km_personal.bin" 


# PHONEME_CATEGORIES = {
#     "vowels": ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"], 
#     "fricatives": ["F", "S", "SH", "TH", "V", "Z", "ZH"], 
#     "nasals": ["M", "N", "NG"], 
#     "affricates": ["CH", "JH"], 
#     "plosives": ["B", "D", "DH", "G", "K", "P", "T", "HH"], 
# }

PHONEME_CATEGORIES = {
    "bilabial": ["B", "P", "M"], 
    "labiodental": ["F", "V"], 
    "dental": ["TH", "DH"], 
    "alveolar": ["T", "D", "S", "Z", "N", "L"], 
    "palatal": ["SH", "ZH", "CH", "JH", "Y"], 
    "velar": ["K", "G", "NG"], 
    "glottal": ["HH", "AH", "R"], 
}

# Map phonemes to their categories
def map_phonemes_to_categories(phonemes, phoneme_categories):
    phoneme_to_category = {}
    for category, phoneme_list in phoneme_categories.items():
        for phoneme in phoneme_list:
            phoneme_to_category[phoneme] = category
    return [phoneme_to_category.get(phoneme, "Unknown") for phoneme in phonemes]

if __name__ == "__main__":

    # Initialize aggregated lists for all phonemes and features
    all_phonemes_unstressed = []
    all_features = []


    for speaker in SPEAKERS:
        speaker_data_path = osp.join(ROOT_DIR, f"{speaker}_data.pkl")
        assert osp.exists(speaker_data_path), f"{speaker_data_path = }"

        with open(speaker_data_path, "rb") as f:
            speaker_data = pickle.load(f)

        print(f"doing speaker {speaker}, {len(speaker_data) = }")
        for recording_data in speaker_data:
            # print(f"{recording_data.keys() = }")
            features = recording_data["features"] 
            phonemes = recording_data["phoneme_ground_truth"]

            phonemes_unstressed = []
            for phoneme in phonemes:
                assert len(phoneme) <= 3
                if len(phoneme) == 3:
                    phonemes_unstressed.append(phoneme[:2])
                else:
                    phonemes_unstressed.append(phoneme)

            # Append the data for this recording to the aggregated lists
            non_empty_ids = [phoneme_idx for phoneme_idx in range(len(phonemes_unstressed)) if phonemes_unstressed[phoneme_idx] != ""] 
            all_phonemes_unstressed.extend([phoneme for (i, phoneme) in enumerate(phonemes_unstressed) if i in non_empty_ids]) 
            all_features.extend(features[non_empty_ids]) 

    # Ensure the lengths match
    assert len(all_phonemes_unstressed) == len(all_features), "Mismatch in lengths!"
    # Ensure the lengths of features and phonemes match
    assert len(all_features) == len(all_phonemes_unstressed), "Mismatch in lengths!"

    # Map phonemes to categories
    all_phoneme_categories = map_phonemes_to_categories(all_phonemes_unstressed, PHONEME_CATEGORIES)

    # Convert features to a numpy array
    # all_features_array = np.array(all_features)
    all_features_array = torch.stack(all_features, dim=0).cpu().numpy() 

    print(f"{all_features_array.shape = }")
    # Learn a KMeans model with 50 clusters
    print("Fitting KMeans model...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    kmeans.fit(all_features_array)

    # Save the KMeans model to disk
    print(f"Saving KMeans model to {KMEANS_MODEL_PATH}...")
    joblib.dump(kmeans, KMEANS_MODEL_PATH)

    print("KMeans model saved successfully!")
