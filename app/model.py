# model.py

import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# Set the backend to sox_io
torchaudio.set_audio_backend("sox_io")

def get_embeddings(model, processor, audio_file):
    waveform, sample_rate = torchaudio.load(audio_file)
    
    # Ensure waveform is 2D (channels, samples)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    
    # Adjust the shape of inputs to match the expected input of Wav2Vec2Model
    input_values = inputs['input_values']
    
    # Squeeze any unnecessary dimensions
    input_values = input_values.squeeze()
    
    # Ensure input_values is 2D
    if input_values.ndim == 1:
        input_values = input_values.unsqueeze(0)
    
    with torch.no_grad():
        embeddings = model(input_values).last_hidden_state.mean(dim=1)
    
    return embeddings

def train_model(train_df):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    # Assuming some additional training steps here if required
    return model, processor