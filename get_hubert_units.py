import torchaudio
import torch
from transformers import HubertModel
import joblib  # For loading the k-means model 

# Step 1: Load the audio file
def get_hubert_clusters(audio_path): 
    waveform, sample_rate = torchaudio.load(audio_path)

    # Normalize the waveform
    waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono 
    waveform = waveform / waveform.abs().max()
    print(f"{waveform.shape = }")

    # Resample the waveform to 16 kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Step 2: Load HuBERT model
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960") 

    # Step 3: Process the waveform directly with HuBERT
    # The input to HuBERT requires a tensor of shape (batch_size, num_samples) 
    inputs = waveform.squeeze(0).unsqueeze(0)  # Reshape to (1, num_samples) 
    print(f"{inputs.shape = }")

    with torch.no_grad():
        hidden_states = model(inputs).last_hidden_state  # Extract HuBERT features
    print(f"{hidden_states.shape = }")

    # Step 4: Load k-means model (pre-trained)
    kmeans_model_path = "km.bin"
    kmeans_model = joblib.load(kmeans_model_path)

    # Apply k-means clustering on the HuBERT features
    cluster_units = kmeans_model.predict(hidden_states.squeeze(0).numpy())
    print(f"{cluster_units.shape = }")

    print("Clustered Units:", cluster_units) 
    return cluster_units 


audio_path = "150.wav" 
clusters = get_hubert_clusters(audio_path) 