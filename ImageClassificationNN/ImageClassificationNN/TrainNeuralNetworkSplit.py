import os
import random
import shutil

PRESPLIT_PATH = 'data/presplit'
DATA_PATH = 'data'

class Splitter(object):
    def Split():
        data = {}

        path = DATA_PATH + '/' + 'train'
        if os.path.exists(path):
            shutil.rmtree(path)

        path = DATA_PATH + '/' + 'test'
        if os.path.exists(path):
            shutil.rmtree(path)

        path = DATA_PATH + '/' + 'validation'
        if os.path.exists(path):
            shutil.rmtree(path)

        for folder_name in os.listdir(PRESPLIT_PATH):
            data[folder_name] = []
            for file_name in os.listdir(PRESPLIT_PATH + '/' + folder_name):
                dat = {}
                dat['folder_name'] = folder_name
                dat['file_name'] = file_name
                data[folder_name].append(dat)

        data_train = []
        data_test = []
        data_validate = []
        pending_data = []

        for key in data:
            random.shuffle(data[key])
            data_train.append(data[key][0])
            data_test.append(data[key][1])
            data_validate.append(data[key][2])
            for i in range(3, len(data[key])):
                pending_data.append(data[key][i])

        total_length = len(pending_data) + len(data_train) + len(data_test) + len(data_validate)
        training_size = int(total_length * (8 / 10))
        testing_size = int((total_length - training_size) / 2)
        validation_size = int((total_length - training_size) / 2)

        random.shuffle(pending_data)
        i = 0
        while len(data_train) < training_size:
            data_train.append(pending_data[i])
            i += 1

        while len(data_test) < testing_size:
            data_test.append(pending_data[i])
            i += 1

        while len(data_validate) < validation_size:
            data_validate.append(pending_data[i])
            i += 1

        for data in data_train:
            folder_name = data['folder_name']
            file_name = data['file_name']
            source_path = PRESPLIT_PATH + '/' + folder_name
            destination_path = DATA_PATH + '/' + 'train' + '/' + folder_name
            if not os.path.exists(destination_path):
                os.makedirs(destination_path)
            shutil.copy(source_path  + '/' + file_name, destination_path)

        for data in data_test:
            folder_name = data['folder_name']
            file_name = data['file_name']
            source_path = PRESPLIT_PATH + '/' + folder_name
            destination_path = DATA_PATH + '/' + 'test' + '/' + folder_name
            if not os.path.exists(destination_path):
                os.makedirs(destination_path)
            shutil.copy(source_path  + '/' + file_name, destination_path)

        for data in data_validate:
            folder_name = data['folder_name']
            file_name = data['file_name']
            source_path = PRESPLIT_PATH + '/' + folder_name
            destination_path = DATA_PATH + '/' + 'validation' + '/' + folder_name
            if not os.path.exists(destination_path):
                os.makedirs(destination_path)
            shutil.copy(source_path  + '/' + file_name, destination_path)

if __name__ == '__main__':
    Splitter.Split()