from importlib.resources import path
import os
import random
import pathlib
import shutil

def setup_folder_structure(dir_data, dir_train, dir_valid, dir_test, img_classes)->None:
    if not dir_data.exists():  dir_data.mkdir()
    if not dir_train.exists(): dir_train.mkdir()
    if not dir_valid.exists(): dir_valid.mkdir()
    if not dir_test.exists():  dir_test.mkdir()
    
    for cls in img_classes:
        if not dir_train.joinpath(cls).exists(): dir_train.joinpath(cls).mkdir()
        if not dir_valid.joinpath(cls).exists(): dir_valid.joinpath(cls).mkdir()
        if not dir_test.joinpath(cls).exists():  dir_test.joinpath(cls).mkdir()
        
    return

def train_test_validation_split(src_folder: pathlib.PosixPath, class_name: str, dir_data)->dict:
    n_train, n_valid, n_test = 0, 0, 0
    for file in src_folder.iterdir():
        x = random.random()
        tgt_dir = ''
        if x <= pct_train:
            tgt_dir = 'train'
            n_train += 1
        elif x <= pct_train + pct_valid:
            tgt_dir = 'validation'
            n_valid +=1
        else:
            tgt_dir = 'test'
            n_test += 1

        img_name = str(file).split('\\')
        where = str(dir_data)
        where = where.replace('\\', '/')
        shutil.copy(
                src=file,
                dst=f'{str(where)}/{tgt_dir}/{class_name}/{img_name[-1]}'
            )

    return  {
        'source': str(src_folder),
        'target': str(dir_data),
        'n_train': n_train,
        'n_validaiton': n_valid,
        'n_test': n_test
    }

def main():
    img_classe = os.listdir('./Imagens')
    dir_data  = pathlib.Path.cwd().joinpath('data')
    dir_train = dir_data.joinpath('train')
    dir_valid = dir_data.joinpath('validation')
    dir_test  = dir_data.joinpath('test')
    setup_folder_structure(dir_data, dir_train, dir_valid, dir_test, img_classe)
    for i in img_classe:
        train_test_validation_split(src_folder=pathlib.Path.cwd().joinpath('Imagens/'+i), class_name=str(i), dir_data=dir_data)

if __name__ == "__main__":
    pct_train = 0.75
    pct_valid = 0.125
    pct_test = 0.125
    main()