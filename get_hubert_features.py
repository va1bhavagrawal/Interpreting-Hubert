import torchaudio
import torch
from transformers import HubertModel
import joblib  # For loading the k-means model 
from utils import * 
import copy 

import os 
import os.path as osp 
import matplotlib.pyplot as plt 
import numpy as np 
import cv2 

from tqdm import tqdm 
import pickle 

# Step 1: Load the audio file
def get_hubert_features(model, audio_path): 
    waveform, sample_rate = torchaudio.load(audio_path)

    # Normalize the waveform
    waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono 
    waveform = waveform / waveform.abs().max()
    # print(f"{waveform.shape = }")

    # Resample the waveform to 16 kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Step 3: Process the waveform directly with HuBERT
    # The input to HuBERT requires a tensor of shape (batch_size, num_samples) 
    inputs = waveform.squeeze(0).unsqueeze(0)  # Reshape to (1, num_samples) 
    # print(f"{inputs.shape = }")

    with torch.no_grad():
        hidden_states = model(inputs.to(0)).last_hidden_state  # Extract HuBERT features
    # print(f"{hidden_states.shape = }")

    return hidden_states 


CORPUS_DIR = "../pratt3000/vctk-corpus/versions/1/VCTK-Corpus/VCTK-Corpus/corpuses" 
ALIGNED_CORPUS_DIR = CORPUS_DIR.replace("corpuses", "aligned_corpuses") 
# SPEAKERS = ["p376", "p345", "p334", "p300"]  
SPEAKERS = ["p376"] 

if __name__ == "__main__": 
    # audio_path = "150.wav" 
    # clusters = get_hubert_clusters(audio_path) 
    # file_path = "./pratt3000/vctk-corpus/versions/1/VCTK-Corpus/VCTK-Corpus/aligned_corpuses/p376/p376_150.TextGrid"  # Replace with your file path
    # assert osp.exists(file_path)
    assert osp.exists(CORPUS_DIR)  
    assert osp.exists(ALIGNED_CORPUS_DIR) 
    # model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(0)  
    model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to(0)  
    state_dict = torch.load("p376_0.0001/009.pth") 
    new_state_dict = copy.deepcopy(model.state_dict()) 
    for k, v in new_state_dict.items(): 
        assert "hubert." + k in state_dict, f"{k = }"  
        new_state_dict[k] = state_dict["hubert." + k] 
    model.load_state_dict(new_state_dict)  

    all_data = [] 

    for speaker in SPEAKERS: 
        speaker_corpus_dir = osp.join(CORPUS_DIR, speaker) 
        speaker_aligned_corpus_dir = osp.join(ALIGNED_CORPUS_DIR, speaker) 
        assert osp.exists(speaker_corpus_dir), f"{speaker_corpus_dir = }"  
        assert osp.exists(speaker_aligned_corpus_dir), f"{speaker_aligned_corpus_dir = }"  

        with tqdm(total=len(os.listdir(speaker_aligned_corpus_dir))) as pbar: 
            for textgrid_name in os.listdir(speaker_aligned_corpus_dir): 
                textgrid_path = osp.join(speaker_aligned_corpus_dir, textgrid_name) 
                wav_path = textgrid_path.replace("aligned_corpuses", "corpuses").replace("TextGrid", "wav") 
                assert osp.exists(wav_path), f"{wav_path = }" 
                assert osp.exists(textgrid_path), f"{textgrid_path = }"  

                # Parse the TextGrid and compute ground truth
                phoneme_intervals = parse_textgrid(textgrid_path) 
                total_duration = max(interval["xmax"] for interval in phoneme_intervals)  # Determine xmax from the intervals 
                frame_size = 0.025  # 20 ms
                hop_size = 0.020  # 25 ms
                phoneme_ground_truth = generate_phoneme_ground_truth(phoneme_intervals, frame_size, hop_size, total_duration)

                # # Print the frame-wise phoneme ground truth
                # for i, phoneme in enumerate(phoneme_ground_truth):
                #     print(f"Frame {i}: {phoneme}")
                # print(f"{phoneme_ground_truth = }") 

                features = get_hubert_features(model, wav_path).squeeze()    
                assert abs(len(features) - len(phoneme_ground_truth)) <= 2, f"{features.shape = }, {len(phoneme_ground_truth) = }, {wav_path = }"  
                # if len(phoneme_ground_truth) == len(clusters) + 1: 
                #     phoneme_ground_truth = phoneme_ground_truth[:-1] 
                min_length = len(phoneme_ground_truth) if len(phoneme_ground_truth) < len(features) else len(features) 
                features = features[:min_length] 
                phoneme_ground_truth = phoneme_ground_truth[:min_length] 
                data = {
                    "wav_path": wav_path, 
                    "textgrid_path": textgrid_path, 
                    "speaker": speaker, 
                    "features": features, 
                    "phoneme_ground_truth": phoneme_ground_truth 
                }
                all_data.append(data) 
                assert len(features) == len(phoneme_ground_truth), f"{len(features), len(phoneme_ground_truth), wav_path = }"  
                pbar.update(1) 

        # Save the data 
        speaker_data_path = osp.join("speakers_hubert_features_personalized", f"{speaker}_data.pkl") 
        os.makedirs(osp.dirname(speaker_data_path), exist_ok=True) 
        with open(speaker_data_path, "wb") as f: 
            pickle.dump(all_data, f)