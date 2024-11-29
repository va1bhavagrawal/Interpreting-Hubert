import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import copy 

ROOT_DIR = "speakers_hubert_features"
SPEAKERS = ["p376", "p345", "p334", "p300"]


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


def plot_confusion_matrix(): 
    pass 


if __name__ == "__main__": 

    # Initialize aggregated lists for all phonemes and features
    all_phonemes_unstressed = []
    all_features = []


    for speaker in SPEAKERS:
        speaker_data_path = osp.join(ROOT_DIR, f"{speaker}_data.pkl")
        assert osp.exists(speaker_data_path), f"{speaker_data_path = }"

        with open(speaker_data_path, "rb") as f:
            speaker_data = pickle.load(f)

        print(f"{len(speaker_data) = }")
        for recording_data in speaker_data:
            print(f"{recording_data.keys() = }")
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