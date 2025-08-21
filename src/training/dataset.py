from torch.utils.data import Dataset
from scripts.pca import load_pca, apply_pca
import os
import torch

class AudioAWOLDataset(Dataset):
    def __init__(self, data_path, pca_path):
        self.data_path = data_path
        self.pca = load_pca(pca_path)
        self.text_embeddings = []
        self.audio_embeddings = []
        for file in os.listdir(f'{data_path}/text_embeddings'):
            fname, ext = file.split('.')
            self.text_embeddings.append(file)
            self.audio_embeddings.append(f'{fname}.pt')
    
    def __len__(self):
        return len(self.text_embeddings)

    def __getitem__(self, index):
        text_emb_path = f'{self.data_path}/text_embeddings/{self.text_embeddings[index]}'
        audio_emb_path = f'{self.data_path}/audio_embeddings/{self.audio_embeddings[index]}'
        text_embedding = torch.load(text_emb_path).float()
        audio_embedding = torch.load(audio_emb_path).float()

        compr_audio_embed = apply_pca(audio_embedding, self.pca) # (750) -> (64)

        return text_embedding, compr_audio_embed
