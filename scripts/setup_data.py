import os
import shutil
from utils.templates import LABEL_TEMPLATES,METADATA_TEMPLATES
from utils.utils import midi_to_register, note_to_midi, technique_abbr_to_full, DYNAMICS
import random


def group_by_techniques(data_path):
    audio_path = f'{data_path}/audio'
    os.makedirs(audio_path, exist_ok=True)

    for d in os.listdir(f'{data_path}/raw/Strings/Violin/'):
        if 'ord' in d or 'trem' in d or 'pizz' in d:
            shutil.copytree(f'{data_path}/raw/Strings/Violin/{d}', audio_path, dirs_exist_ok=True)

    for d in os.listdir(f'{data_path}/raw/Strings/Violin+sordina/'):
        if 'ord' in d or 'trem' in d or 'pizz' in d:
            shutil.copytree(f'{data_path}/raw/Strings/Violin+sordina/{d}', audio_path, dirs_exist_ok=True)

    for d in os.listdir(f'{data_path}/raw/Strings/Violin+sordina_piombo/'):
        if 'ord' in d or 'trem' in d or 'pizz' in d:
            shutil.copytree(f'{data_path}/raw/Strings/Violin+sordina_piombo/{d}', audio_path, dirs_exist_ok=True)


def generate_label(file_name):
    
    metadata = file_name.split('-')
    technique = technique_abbr_to_full(metadata[1])
    register = midi_to_register(note_to_midi(metadata[2]))
    dynamics = DYNAMICS[metadata[3]]

    metadata_enhs = []
    base_templates = LABEL_TEMPLATES[technique]
    base_label = random.choice(base_templates)
    dynamic_template = random.choice(METADATA_TEMPLATES['dynamic_descriptors'])
    metadata_enhs.append(dynamic_template.format(dynamic=dynamics))
        
    register_template = random.choice(METADATA_TEMPLATES['register_descriptors'])
    metadata_enhs.append(register_template.format(register=register))

    if metadata_enhs:
        random.shuffle(metadata_enhs)
        enhanced_label = base_label + " " + " ".join(metadata_enhs)
    else:
        enhanced_label = base_label
    
    return enhanced_label

def generate_labels(data_path):
    os.makedirs(f'{data_path}/labels', exist_ok=True)

    for file in os.listdir(f'{data_path}/audio'):
        file_name, ext = file.split('.')
        label = generate_label(file_name)

        with open(f'{data_path}/labels/{file_name}.txt', mode='w', encoding='utf-8') as label_file:
            label_file.write(label)


if __name__ == '__main__':
    group_by_techniques('./data')
    generate_labels('./data')