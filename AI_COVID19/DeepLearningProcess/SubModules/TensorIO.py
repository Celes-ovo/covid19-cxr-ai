from AI_COVID19.init import *


# Class Modules
def class_split(dataset, mode):
    """
    :param dataset: Training에 사용될 전체 데이터셋 리스트
    :param mode: '4cls' or '3cls'
    :return:
    """
    if mode == '4cls':
        dataset_dic = {'Covid19': [], 'Bacterial': [], 'Viral': [], 'Normal': []}
        for i, i_ in enumerate(dataset):
            if i_['info']['cls_label'] == 'Covid19':
                i_['info'].update({'onehot': [1, 0, 0, 0]})
                dataset_dic['Covid19'].append(i_)
            elif i_['info']['cls_label'] == 'Bacterial':
                i_['info'].update({'onehot': [0, 1, 0, 0]})
                dataset_dic['Bacterial'].append(i_)
            elif i_['info']['cls_label'] == 'Viral':
                i_['info'].update({'onehot': [0, 0, 1, 0]})
                dataset_dic['Viral'].append(i_)
            elif i_['info']['cls_label'] == 'Normal':
                i_['info'].update({'onehot': [0, 0, 0, 1]})
                dataset_dic['Normal'].append(i_)
        return dataset_dic
    elif mode == '3cls':
        dataset_dic = {'Covid19': [], 'Pneumonia': [], 'Normal': []}
        for i, i_ in enumerate(dataset):
            if i_['info']['cls_label'] == 'Covid19':
                i_['info'].update({'onehot': [1, 0, 0]})
                dataset_dic['Covid19'].append(i_)
            elif i_['info']['cls_label'] == 'Bacterial':
                i_['info'].update({'onehot': [0, 1, 0]})
                dataset_dic['Pneumonia'].append(i_)
            elif i_['info']['cls_label'] == 'Viral':
                i_['info'].update({'onehot': [0, 1, 0]})
                dataset_dic['Pneumonia'].append(i_)
            elif i_['info']['cls_label'] == 'Normal':
                i_['info'].update({'onehot': [0, 0, 1]})
                dataset_dic['Normal'].append(i_)
        return dataset_dic


def class_merging(tensor_list):
    for i, i_ in enumerate(tensor_list):
        if i == 0:
            pass
        elif i == 1:
            train_x = np.concatenate(i_['Train'][0], tensor_list[i-1]['Train'][0], axis=0)
            train_y = np.concatenate(i_['Train'][1], tensor_list[i-1]['Train'][1], axis=0)
        else:
            train_x = np.concatenate(train_x, i_['Train'][0], axis=0)
            train_y = np.concatenate(train_y, i_['Train'][0], axis=0)
    return train_x, train_y


# Train-Test Split
def train_test_split(dataset, split_rate, mode):
    dataset_dic = {}
    for i, i_ in enumerate(list(dataset.keys())):
        dataset_dic.update({i_: {'Train': [], 'Val': [], 'Test': []}})
        if mode == 'rate':
            test_num = int(len(dataset[i_]) * split_rate['test'])
            val_num = int(len(dataset[i_]) * split_rate['val'])
        elif mode == 'num':
            test_num = int(split_rate['test'])
            val_num = int(split_rate['val'])
        if test_num == 0:
            test_num = 1
        if val_num == 0:
            val_num = 1
        for j, j_ in enumerate(dataset[i_]):
            if j < test_num:
                dataset[i_][j]['info'].update({'set': 'Test'})
                dataset_dic[i_]['Test'].append(dataset[i_][j])
            else:
                if test_num <= j < test_num + val_num:
                    dataset[i_][j]['info'].update({'set': 'Val'})
                    dataset_dic[i_]['Val'].append(dataset[i_][j])
                else:
                    dataset[i_][j]['info'].update({'set': 'Train'})
                    dataset_dic[i_]['Train'].append(dataset[i_][j])
    return dataset_dic


def train_test_split_kfold(dataset, split_rate):
    pass


def tensor_setting(dataset):
    x_list = []
    y_list = []
    for i, i_ in enumerate(dataset):
        x_list.append(i_['image'])
        y_list.append(i_['info']['onehot'])
    return np.expand_dims(np.array(x_list, dtype=np.uint8), axis=-1), np.array(y_list, dtype=np.uint8)




