import os
import shutil

def group_by_techniques(data_path):
    ordinario_path = f'{data_path}/audio/ordinario'
    tremolo_path = f'{data_path}/audio/tremolo'
    pizzicato_path = f'{data_path}/audio/pizzicato'
    os.makedirs(ordinario_path, exist_ok=True)
    os.makedirs(tremolo_path, exist_ok=True)
    os.makedirs(pizzicato_path, exist_ok=True)

    for d in os.listdir(f'{data_path}/raw/Strings/Violin/'):
        if 'ordinario' in d or 'sforzato' in d:
            shutil.copytree(f'{data_path}/raw/Strings/Violin/{d}', ordinario_path, dirs_exist_ok=True)
        elif 'tremolo' in d:
            shutil.copytree(f'{data_path}/raw/Strings/Violin/{d}', tremolo_path, dirs_exist_ok=True)
        elif 'pizzicato' in d:
            shutil.copytree(f'{data_path}/raw/Strings/Violin/{d}', pizzicato_path, dirs_exist_ok=True)
    

    for d in os.listdir(f'{data_path}/raw/Strings/Violin+sordina/'):
        if 'ordinario' in d or 'sforzato' in d:
            shutil.copytree(f'{data_path}/raw/Strings/Violin+sordina/{d}', ordinario_path, dirs_exist_ok=True)
        elif 'tremolo' in d:
            shutil.copytree(f'{data_path}/raw/Strings/Violin+sordina/{d}', tremolo_path, dirs_exist_ok=True)
        elif 'pizzicato' in d:
            shutil.copytree(f'{data_path}/raw/Strings/Violin+sordina/{d}', pizzicato_path, dirs_exist_ok=True)
    

    for d in os.listdir(f'{data_path}/raw/Strings/Violin+sordina_piombo/'):
        if 'ordinario' in d or 'sforzato' in d:
            shutil.copytree(f'{data_path}/raw/Strings/Violin+sordina_piombo/{d}', ordinario_path, dirs_exist_ok=True)
        elif 'tremolo' in d:
            shutil.copytree(f'{data_path}/raw/Strings/Violin+sordina_piombo/{d}', tremolo_path, dirs_exist_ok=True)
        elif 'pizzicato' in d:
            shutil.copytree(f'{data_path}/raw/Strings/Violin+sordina_piombo/{d}', pizzicato_path, dirs_exist_ok=True)


if __name__ == '__main__':
    group_by_techniques('./data')