import pandas as pd
import torchaudio
from torchaudio.transforms import MelSpectrogram

def load_data():
    train_df = pd.read_csv('Dataset/train.csv')
    database_df = pd.read_csv('Dataset/database.csv')
    test_df = pd.read_csv('Dataset/test.csv')
    return train_df, database_df, test_df

def extract_features(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    mel_spectrogram = MelSpectrogram()(waveform)
    return mel_spectrogram
