import torch

NOTES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
NOTES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
DYNAMICS = {'pp': 'pianissimo',
            'p': 'piano',
            'mf': 'mezzoforte',
            'fp': 'fortepiano',
            'f': 'forte',
            'ff':'fortissimo'}

def note_to_midi(key_octave):
    key = key_octave[:-1]  # eg C, Db
    octave = key_octave[-1]   # eg 3, 4
    answer = -1

    try:
        if 'b' in key:
            pos = NOTES_FLAT.index(key)
        else:
            pos = NOTES_SHARP.index(key)
    except:
        print('The key is not valid', key)
        return answer

    answer += pos + 12 * (int(octave) + 1) + 1
    return answer

def midi_to_register(midi):
    if midi <= 67:
        return 'low'
    elif midi > 67 and midi <= 83:
        return 'middle'
    else:
        return 'high'

def technique_abbr_to_full(abbr):
    if 'ord' in abbr or 'sfz' in abbr:
        return 'standard_bowed'
    elif 'trem' in abbr:
        return 'tremolo'
    elif 'pizz' in abbr:
        return 'pizzicato'


def load_embedding_pt(path):
    embedding = torch.load(path, map_location="cpu")    
    return embedding


def split_embedding(embedding, n_frames):
    f0_hz = embedding[0:n_frames]
    f0_conf = embedding[n_frames:2*n_frames]
    loudness_db = embedding[2*n_frames:3*n_frames]
    return f0_hz, f0_conf, loudness_db

def pad_embedding(embedding, n_frames):
    f0_hz, f0_conf, loudness_db = split_embedding(embedding, embedding.shape[0]//3)

    padding_needed = n_frames - len(f0_hz)
    f0_pad = torch.zeros(padding_needed)
    loudness_pad = torch.tensor([-80]*padding_needed)

    f0_hz, f0_conf, loudness_db = torch.concat((f0_hz, f0_pad)), \
                                    torch.concat((f0_conf, f0_pad)), \
                                    torch.concat((loudness_db, loudness_pad))
    x = torch.concat((f0_hz, f0_conf, loudness_db))

    return x

