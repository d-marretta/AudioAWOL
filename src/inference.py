import ddsp.training
import numpy as np
from utils.utils import split_embedding
import tensorflow as tf
import gin
import ddsp
import os

def load_ddsp(ddsp_path):
    gin_file = os.path.join(ddsp_path, 'operative_config-0.gin')
    gin.parse_config_file(gin_file, skip_unknown=True)
    gin.bind_parameter('F0LoudnessPreprocessor.compute_loudness', False)

    # Build model by calling it once with dummy features
    model = ddsp.training.models.Autoencoder()

    # Restore checkpoint of pretrained violin model
    ckpt = tf.train.Checkpoint(model=model)
    status = ckpt.restore(f'{ddsp_path}/ckpt-40000')
    status.expect_partial() 
    dummy_features = {
        'f0_hz': tf.zeros([1, 100], dtype=tf.float32),
        'f0_confidence': tf.zeros([1, 100], dtype=tf.float32),
        'loudness_db': tf.zeros([1, 100], dtype=tf.float32),
        'audio': tf.zeros([1, 16000], dtype=tf.float32),
    }
    _ = model(dummy_features, training=False)

    return model

def generate_audio_from_features(model, embedding, n_frames, sample_rate=16000):
    # Split the concatenated vector back into features and repeat for four times,
    # since the model was trained with 1000 frames as input
    f0_hz, f0_conf, loudness_db = split_embedding(embedding, n_frames)
    f0_hz_rep = np.tile(f0_hz, 4)
    f0_conf_rep = np.tile(f0_conf, 4)
    loudness_db_rep = np.tile(loudness_db, 4)

    # Add batch dimension: shape [1, time]
    f0_hz = tf.constant(f0_hz_rep[None, :], dtype=tf.float32)
    f0_conf = tf.constant(f0_conf_rep[None, :], dtype=tf.float32)
    loudness_db = tf.constant(loudness_db_rep[None, :], dtype=tf.float32)

    features = {
        'f0_hz': f0_hz,
        'f0_confidence': f0_conf,
        'loudness_db': loudness_db,
        'audio': tf.zeros([1, sample_rate], dtype=tf.float32),  # 1 second dummy audio
    }
    outputs = model(features, training=False)
    return outputs['audio_synth'].numpy().squeeze()[:sample_rate]
