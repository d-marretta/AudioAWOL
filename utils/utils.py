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
    n = n_frames
    f0_hz = embedding[0:n]
    f0_conf = embedding[n:2*n]
    loudness_db = embedding[2*n:3*n]
    return f0_hz, f0_conf, loudness_db
