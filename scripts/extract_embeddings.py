from transformers import ClapProcessor, ClapModel
import torch
import os

def get_labels(labels_path):
    # get list of file names and labels
    names = []
    labels = []
    for file in os.listdir(labels_path):
        file_name, ext = file.split('.')
        with open(f'{labels_path}/{file}', mode='r', encoding='utf-8') as label_file:
            label = label_file.read()
            labels.append(label)
            names.append(file_name)
    
    return names, labels
    

def extract_text_embeddings(data_path):

    model = ClapModel.from_pretrained("laion/larger_clap_music")
    processor = ClapProcessor.from_pretrained("laion/larger_clap_music")

    model.eval()

    names, labels = get_labels(f'{data_path}/labels')

    inputs = processor(text=labels, return_tensors="pt", padding=True)

    with torch.no_grad():
        text_embeddings = model.get_text_features(**inputs)

    # save embeddings
    os.makedirs(f'{data_path}/text_embeddings', exist_ok=True)
    for i, embedding in enumerate(text_embeddings):
        torch.save(embedding, f'{data_path}/text_embeddings/{names[i]}.pt')
        


if __name__ == '__main__':
    extract_text_embeddings('./data')