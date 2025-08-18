import os
import torch
import numpy as np
from sklearn.decomposition import PCA
import joblib
import yaml

def fit_pca(pca_path, data_path, n_components):
    os.makedirs(pca_path, exist_ok=True)

    all_tensors = []
    for fname in os.listdir(f'{data_path}/audio_embeddings'):
        if fname.endswith(".pt"):
            x = torch.load(os.path.join(f'{data_path}/audio_embeddings', fname))

            x = x.view(x.shape[0], -1) if x.ndim > 1 else x.unsqueeze(0)
            if x.shape[1] != 750:
                # TODO: implement padding (loudness min -80)
                pass
            else:
                all_tensors.append(x.numpy())
    
    data = np.vstack(all_tensors)  # shape: (total_samples, 750)

    pca = PCA(n_components=n_components)
    pca = pca.fit(data)   # shape: (total_samples, N_COMPONENTS)

    joblib.dump(pca, os.path.join(pca_path, 'pca.z'))

def load_pca(pca_path):
    pca = joblib.load(os.path.join(pca_path, 'pca.z'))
    return pca

def apply_pca(x, pca_path):
    pca = load_pca(pca_path)
    z = pca.transform(x.numpy())
    return torch.from_numpy(z).float()

def inverse_pca(z, pca_path):
    pca = load_pca(pca_path)
    x_recon = pca.inverse_transform(z.numpy())
    return torch.from_numpy(x_recon).float()


if __name__ == '__main__':
    with open('./configs/config.yaml', mode='r') as config_file:
        config = yaml.load(config_file, yaml.Loader)
    pca_path = config['PCA_PATH']
    n_components = config['PCA_COMPONENTS']
    data_path = config['DATASET_DIR']
    fit_pca(pca_path, data_path, n_components)