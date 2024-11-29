import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import copy 

ROOT_DIR = "speakers_hubert_data"
SPEAKERS = ["p376", "p345", "p334", "p300"]


PHONEME_CATEGORIES = {
    "vowels": ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"], 
    "fricatives": ["F", "S", "SH", "TH", "V", "Z", "ZH"], 
    "nasals": ["M", "N", "NG"], 
    "affricates": ["CH", "JH"], 
    "plosives": ["P", "B", "T", "D", "K", "G"],  
}

# PHONEME_CATEGORIES = {
#     "bilabial": ["B", "P", "M"], 
#     "labiodental": ["F", "V"], 
#     "dental": ["TH", "DH"], 
#     "alveolar": ["T", "D", "S", "Z", "N", "L"], 
#     "palatal": ["SH", "ZH", "CH", "JH", "Y"], 
#     "velar": ["K", "G", "NG"], 
#     "glottal": ["HH", "AH", "R"], 
# }


if __name__ == "__main__": 

    # Initialize aggregated lists for all phonemes and clusters
    all_phonemes_unstressed = []
    all_clusters = []


    for speaker in SPEAKERS:
        speaker_data_path = osp.join(ROOT_DIR, f"{speaker}_data.pkl")
        assert osp.exists(speaker_data_path), f"{speaker_data_path = }"

        with open(speaker_data_path, "rb") as f:
            speaker_data = pickle.load(f)

        print(f"{len(speaker_data) = }")
        for recording_data in speaker_data:
            print(f"{recording_data.keys() = }")
            clusters = recording_data["clusters"]
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
            all_clusters.extend(clusters[non_empty_ids]) 

    # Ensure the lengths match
    assert len(all_phonemes_unstressed) == len(all_clusters), "Mismatch in lengths!"
    all_phonemes_unstressed_ = copy.deepcopy(all_phonemes_unstressed) 
    all_clusters_ = copy.deepcopy(all_clusters) 

    for phoneme_category in PHONEME_CATEGORIES: 
        # Map phonemes and clusters to integer labels
        all_phonemes_unstressed = copy.deepcopy(all_phonemes_unstressed_) 
        all_clusters = copy.deepcopy(all_clusters_) 

        category_ids = [i for i in range(len(all_phonemes_unstressed)) if all_phonemes_unstressed[i] in PHONEME_CATEGORIES[phoneme_category]] 
        all_phonemes_unstressed = [all_phonemes_unstressed[i] for i in category_ids] 
        all_clusters = [all_clusters[i] for i in category_ids] 

        unique_phonemes = sorted(set(all_phonemes_unstressed)) 
        unique_clusters = sorted(set(all_clusters)) 

        phoneme_to_idx = {phoneme: i for i, phoneme in enumerate(unique_phonemes)} 
        cluster_to_idx = {cluster: i for i, cluster in enumerate(unique_clusters)} 
        print(f"{len(phoneme_to_idx), len(cluster_to_idx) = }")  

        phoneme_labels = np.array([phoneme_to_idx[phoneme] for phoneme in all_phonemes_unstressed]).astype(np.int32) 
        cluster_labels = np.array([cluster_to_idx[cluster] for cluster in all_clusters]).astype(np.int32) 
        print(f"{len(np.unique(phoneme_labels)) = }, {len(np.unique(cluster_labels)) = }") 

        confusion_matrix = np.zeros((len(unique_phonemes), len(unique_clusters))) 
        for i in range(len(phoneme_labels)): 
            confusion_matrix[phoneme_labels[i], cluster_labels[i]] += 1 

        # plt.imshow(confusion_matrix, cmap="viridis") 
        # plt.colorbar() 
        # plt.savefig("plot.jpg") 
        # Plot the confusion matrix with phoneme and cluster labels
        plt.figure(figsize=(15, 10))  # Adjust figure size for better readability
        plt.imshow(confusion_matrix, cmap="viridis")
        plt.colorbar()

        # Set axis labels
        plt.xlabel("Clusters", fontsize=14)
        plt.ylabel("Phonemes", fontsize=14)

        # Set axis tick labels
        plt.xticks(ticks=np.arange(len(unique_clusters)), labels=unique_clusters, rotation=90, fontsize=10)
        plt.yticks(ticks=np.arange(len(unique_phonemes)), labels=unique_phonemes, fontsize=10)

        # Save the plot
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig(f"{len(SPEAKERS)}speaker__{phoneme_category}.jpg") 