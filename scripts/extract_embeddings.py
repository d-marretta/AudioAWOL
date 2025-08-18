from transformers import ClapProcessor, ClapModel
import torch
import os
import ddsp
import ddsp.training
import warnings
import yaml
import tensorflow as tf
import soundfile as sf
import numpy as np
import librosa
import gin
from sklearn.decomposition import PCA
import joblib
from utils.utils import load_embedding_pt
from src.inference import generate_audio_from_features, load_ddsp

def get_labels(labels_path):
    # Get list of file names and labels
    names = []
    labels = []
    for file in os.listdir(labels_path):
        file_name, ext = file.split('.')
        with open(f'{labels_path}/{file}', mode='r', encoding='utf-8') as label_file:
            label = label_file.read()
            labels.append(label)
            names.append(file_name)
    
    return names, labels
    

def get_text_embeddings(data_path):
    model = ClapModel.from_pretrained("laion/larger_clap_music")
    processor = ClapProcessor.from_pretrained("laion/larger_clap_music")

    model.eval()

    names, labels = get_labels(f'{data_path}/labels')

    inputs = processor(text=labels, return_tensors="pt", padding=True)

    with torch.no_grad():
        text_embeddings = model.get_text_features(**inputs)

    # Save embeddings
    os.makedirs(f'{data_path}/text_embeddings', exist_ok=True)
    for i, embedding in enumerate(text_embeddings):
        torch.save(embedding, f'{data_path}/text_embeddings/{names[i]}.pt')
    

def get_audio_embedding(audio_file, sample_rate):
    audio, sr = sf.read(audio_file)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # mono

    # If file's sample rate doesn't match model's sample rate, resample
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate 
    
    # Take only the first second
    num_samples = sample_rate  # since sample_rate (16000) samples = 1 second
    if len(audio) > num_samples:
        audio = audio[:num_samples-1]
    
    audio = audio.astype(np.float32)
    
    # Prepare features (loudness_db, f0_hz, f0_confidence)
    features = ddsp.training.metrics.compute_audio_features(audio)
    f0_hz = features['f0_hz']
    f0_conf = features['f0_confidence']
    loudness_db = features['loudness_db']   

    embedding = np.concatenate([f0_hz, f0_conf, loudness_db], axis=0)  

    return embedding

def get_audio_embeddings(data_path, sample_rate):
    os.makedirs(f'{data_path}/audio_embeddings', exist_ok=True)
    for file in os.listdir(f'{data_path}/audio'):
        file_name, ext = file.split('.')

        audio_embedding = get_audio_embedding(f'{data_path}/audio/{file}', sample_rate)

        # Save all embeddings using torch for consistency with text embeddings
        torch.save(torch.from_numpy(audio_embedding), f'{data_path}/audio_embeddings/{file_name}.pt')


if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="divide by zero encountered in log10")
    
    with open('./configs/config.yaml', mode='r') as config_file:
        config = yaml.load(config_file, yaml.Loader)
    ddsp_path = config['DDSP_MODEL_DIR']
    data_path = config['DATASET_DIR']
    sample_rate = config['DDSP_SAMPLE_RATE']
    get_text_embeddings(data_path)
    get_audio_embeddings(data_path, sample_rate)
