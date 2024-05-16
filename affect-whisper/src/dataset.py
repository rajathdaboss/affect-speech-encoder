import os
import pandas as pd
import soundfile as sf
from scipy.signal import resample
import torch
from transformers import WhisperFeatureExtractor


class WTCAudioDataset(torch.utils.data.Dataset):
    def __init__(self, emb_space='affect'):
        self.emb_space = emb_space
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained('openai/whisper-small', cache_dir='/data/rrao/pretrained/cache/')
        self.df = pd.read_csv(f'/data/rrao/wtc_clinic/affects/clinic_audio_segment_{self.emb_space}_g.csv')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        audio_path = os.path.join(f'/data/WTC/Clinic_Audio_Files/Batch{self.df.iloc[index]['batch']}', self.df.iloc[index]['filename'])
        audio_data, sample_rate = sf.read(audio_path)
        start_index = int(self.df.iloc[index]['start'] * sample_rate)
        end_index = int(self.df.iloc[index]['end'] * sample_rate)
        audio_data = audio_data[start_index:end_index]
        audio_data = resample(audio_data, int(len(audio_data) * 16000 / sample_rate))
        audio_data = torch.tensor(audio_data, dtype=torch.float32)
        log_mel = self.feature_extractor(audio_data, sampling_rate=16000, return_tensors='pt').input_features.to(torch.float32)
        embedding = torch.tensor([e for e in self.df.iloc[index].values[5:]], dtype=torch.float32)
        return log_mel, embedding


def collate(batch):
    return {
        'inputs': torch.cat([segment for segment, _ in batch], dim=0),
        'embeddings': torch.stack([embedding for _, embedding in batch], dim=0)
    }