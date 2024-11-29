import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict
import torch 

import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import copy 

ROOT_DIR = "speakers_hubert_features"
SPEAKERS = ["p376"] 


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
    # "glottal": ["HH"], 
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

    # Apply t-SNE
    print(f"making tsne...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=50)
    reduced_features = tsne.fit_transform(all_features_array)

    # Plot the t-SNE visualization
    plt.figure(figsize=(10, 8))
    categories = list(PHONEME_CATEGORIES.keys())
    category_to_color = {category: plt.cm.tab10(i / len(categories)) for i, category in enumerate(categories)}

    # for category in categories:
    #     indices = [i for i, cat in enumerate(all_phoneme_categories) if cat == category]
    #     plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1],
    #                 label=category, color=category_to_color[category], alpha=1.0) 

    # Assume 'categories' is a list of unique categories
    # Assign a unique color from a colormap for each category
    category_to_color = {category: plt.cm.tab10(i / len(categories)) for i, category in enumerate(categories)}

    for category in categories:
        indices = [i for i, cat in enumerate(all_phoneme_categories) if cat == category]
        plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1],
                    label=category, color=category_to_color[category], alpha=1.0)


    plt.legend()
    plt.title("t-SNE Visualization of Phoneme Categories")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig("tsne.jpg") 
