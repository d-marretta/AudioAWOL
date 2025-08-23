import os
import torch
import numpy as np
from sklearn.decomposition import PCA
import joblib
import yaml
from utils.utils import pad_embedding

def fit_pca(pca_path, data_path, n_components, n_frames):
    os.makedirs(pca_path, exist_ok=True)

    all_tensors = []
    for fname in os.listdir(f'{data_path}/audio_embeddings'):
        if fname.endswith(".pt"):
            x = torch.load(os.path.join(f'{data_path}/audio_embeddings', fname))
            x = x.view(x.shape[0], -1) if x.ndim > 1 else x.unsqueeze(0)

            if x.shape[1] != n_frames*3:
                # If some files are shorter than 1 second, pad the features with 0 for the f0_hz and f0_conf
                # and -80 for the loudness (-80 was the min in all the dataset so it's silence)
                x = pad_embedding(x.squeeze(0), n_frames)
                all_tensors.append(x.unsqueeze(0).numpy())
            else:
                all_tensors.append(x.numpy())
    
    data = np.vstack(all_tensors)  # shape: (total_samples, 750)

    pca = PCA(n_components=n_components)
    pca = pca.fit(data)   # shape: (total_samples, N_COMPONENTS)

    joblib.dump(pca, os.path.join(pca_path, 'pca.z'))

def load_pca(pca_path):
    pca = joblib.load(os.path.join(pca_path, 'pca.z'))
    return pca

def apply_pca(x, pca):
    z = pca.transform(x.numpy().reshape(1,-1))
    return torch.from_numpy(z.squeeze(0)).float()

def inverse_pca(z, pca):
    x_recon = pca.inverse_transform(z.numpy().reshape(1, -1))  
    return torch.from_numpy(x_recon.squeeze(0)).float()


if __name__ == '__main__':
    with open('./configs/config.yaml', mode='r') as config_file:
        config = yaml.load(config_file, yaml.Loader)
    pca_path = config['PCA_PATH']
    n_components = config['PCA_COMPONENTS']
    data_path = config['DATASET_DIR']
    fit_pca(pca_path, data_path, n_components, 250)