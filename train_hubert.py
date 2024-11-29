from transformers import AutoProcessor, HubertForCTC
from datasets import load_dataset
import torch
import torchaudio 

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation", trust_remote_code=True)
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate 

processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

# audio file is decoded on the fly
# print(f"{dataset[0]['audio'] = }") 
# print(f"{dataset[0]['audio']['array'].shape = }") 
# inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt") 

waveform, sample_rate = torchaudio.load("sample_audio.wav")

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
# inputs = waveform.squeeze(0).unsqueeze(0)  # Reshape to (1, num_samples) 

print(f"{waveform.shape = }")
inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt") 
with torch.no_grad():
    logits = model(**inputs).logits
print(f"{logits.shape = }")
predicted_ids = torch.argmax(logits, dim=-1)

# transcribe speech
transcription = processor.batch_decode(predicted_ids)
print(f"{transcription[0] = }") 

# what if you want to compute the los in house 
inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="pt").input_ids

# compute loss
loss = model(**inputs).loss
print(f"{loss = }")