import torchaudio
from transformers import AutoProcessor, HubertForCTC
import torch
from transformers import HubertModel
import joblib  # For loading the k-means model 
from utils import * 

import os 
import os.path as osp 
import matplotlib.pyplot as plt 
import numpy as np 
import cv2 

from tqdm import tqdm 
import pickle 

import wandb 

CORPUS_DIR = "../pratt3000/vctk-corpus/versions/1/VCTK-Corpus/VCTK-Corpus/corpuses" 
ALIGNED_CORPUS_DIR = CORPUS_DIR.replace("corpuses", "aligned_corpuses") 
SPEAKERS = ["p376"] 

LR = 1e-3            
NUM_EPOCHS = 10  

if __name__ == "__main__": 
    # audio_path = "150.wav" 
    # clusters = get_hubert_clusters(audio_path) 
    # file_path = "./pratt3000/vctk-corpus/versions/1/VCTK-Corpus/VCTK-Corpus/aligned_corpuses/p376/p376_150.TextGrid"  # Replace with your file path
    # assert osp.exists(file_path)
    wandb.login(key="6ab81b60046f7d7f6a7dca014a2fcaf4538ff14a") 

    for speaker in SPEAKERS: 
        assert osp.exists(CORPUS_DIR)  
        assert osp.exists(ALIGNED_CORPUS_DIR) 
        # model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(0)  
        processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft") 
        model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(0) 
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR) 

        all_data = [] 
        processed_inputs = [] 
        run_name = f"{speaker}_{LR}"
        wandb.init(project="interpreting_hubert", name=run_name)   
        speaker_corpus_dir = osp.join(CORPUS_DIR, speaker) 
        speaker_aligned_corpus_dir = osp.join(ALIGNED_CORPUS_DIR, speaker) 

        assert osp.exists(speaker_corpus_dir), f"{speaker_corpus_dir = }"  
        assert osp.exists(speaker_aligned_corpus_dir), f"{speaker_aligned_corpus_dir = }"  

        for epoch in range(NUM_EPOCHS): 
            with tqdm(total=len(os.listdir(speaker_aligned_corpus_dir)), desc=f"epoch {epoch + 1} / {NUM_EPOCHS}") as pbar: 
                for textgrid_name in os.listdir(speaker_aligned_corpus_dir): 
                    textgrid_path = osp.join(speaker_aligned_corpus_dir, textgrid_name) 
                    wav_path = textgrid_path.replace("aligned_corpuses", "corpuses").replace("TextGrid", "wav") 
                    txt_path = textgrid_path.replace("aligned_corpuses", "corpuses").replace("TextGrid", "txt") 
                    assert osp.exists(wav_path), f"{wav_path = }" 
                    assert osp.exists(textgrid_path), f"{textgrid_path = }"  
                    assert osp.exists(txt_path), f"{txt_path = }"  

                    # # Parse the TextGrid and compute ground truth
                    # phoneme_intervals = parse_textgrid(textgrid_path) 
                    # total_duration = max(interval["xmax"] for interval in phoneme_intervals)  # Determine xmax from the intervals 
                    # frame_size = 0.025  # 20 ms
                    # hop_size = 0.020  # 25 ms
                    # phoneme_ground_truth = generate_phoneme_ground_truth(phoneme_intervals, frame_size, hop_size, total_duration)

                    # # Print the frame-wise phoneme ground truth
                    # for i, phoneme in enumerate(phoneme_ground_truth):
                    #     print(f"Frame {i}: {phoneme}")
                    # print(f"{phoneme_ground_truth = }") 

                    # if len(phoneme_ground_truth) == len(clusters) + 1: 
                    #     phoneme_ground_truth = phoneme_ground_truth[:-1] 
                    # min_length = len(phoneme_ground_truth) if len(phoneme_ground_truth) < len(clusters) else len(clusters) 
                    # clusters = clusters[:min_length] 
                    # phoneme_ground_truth = phoneme_ground_truth[:min_length] 
                    # data = {
                    #     "wav_path": wav_path, 
                    #     "textgrid_path": textgrid_path, 
                    #     "speaker": speaker, 
                    #     "clusters": clusters, 
                    #     "phoneme_ground_truth": phoneme_ground_truth 
                    # }
                    # all_data.append(data) 
                    # assert len(clusters) == len(phoneme_ground_truth), f"{len(clusters), len(phoneme_ground_truth), wav_path = }"  


                    waveform, sample_rate = torchaudio.load(wav_path)
                    # Normalize the waveform
                    waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono 
                    waveform = waveform / waveform.abs().max()
                    # print(f"{waveform.shape = }")
                    # Resample the waveform to 16 kHz if needed
                    if sample_rate != 16000:
                        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                        waveform = resampler(waveform)
                    # print(f"{waveform.shape = }")
                    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt") 
                    # with torch.no_grad():
                    #     logits = model(**inputs).logits
                    # print(f"{logits.shape = }")
                    # predicted_ids = torch.argmax(logits, dim=-1)

                    # # transcribe speech
                    # transcription = processor.batch_decode(predicted_ids)
                    # print(f"{transcription[0] = }") 

                    # what if you want to compute the los in house 
                    with open(txt_path, "r") as f: 
                        text = f.read().strip() 
                    # inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="pt").input_ids
                    inputs["labels"] = processor(text=text, return_tensors="pt").input_ids 
                    for k, v in inputs.items(): 
                        inputs[k] = v.to(0) 
                    processed_inputs.append(inputs) 

                    loss = model(**inputs).loss 
                    # print(f"{loss = }") 
                    wandb.log({"CTC loss": loss.item()}) 
                    loss.backward() 
                    optimizer.step() 
                    optimizer.zero_grad() 

                    pbar.update(1) 

            if (epoch + 1) % 5 == 0: 
                save_dir = osp.join(run_name) 
                os.makedirs(save_dir, exist_ok=True) 
                torch.save(model.state_dict(), (osp.join(save_dir, f"{str(epoch).zfill(3)}.pth")))  

            # # Save the data 
            # speaker_data_path = osp.join("speakers_hubert_data", f"{speaker}_data.pkl") 
            # os.makedirs(osp.dirname(speaker_data_path)) 
            # with open(speaker_data_path, "wb") as f: 
            #     pickle.dump(all_data, f)
