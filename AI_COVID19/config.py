from AI_COVID19.init import *

path_title = {'local': ['p211001', 't211001'], # project folder title ([전처리 결과 폴더, 학습 결과 폴더])
              'server': ['p211001', 't211001'],
              'test': 'r200730',
              'tests': 'r200731'}

main_path = {'local': 'C:\\Users\\Owner3\\Dataset\\COVID19', # local 환경 경로
             'server':  '/home/jpulsar/Dataset/COVID19'} # server 환경 경로

path_param_set = {'local': {'input_path': {'dataset': os.path.join(main_path['local'], 'new_data'),
                                           'prep': os.path.join(main_path['local'], '%s\\org' % path_title['local'][0]),
                                           'train': os.path.join(main_path['local'], '%s\\prep' % path_title['local'][0])},
                             'output_path': {'log': os.path.join(main_path['local'], 'log'),
                                             'prep': os.path.join(main_path['local'], '%s' % path_title['local'][0]),
                                             'train': os.path.join(main_path['local'], '%s' % path_title['local'][1])}},
                  'server': {'input_path': {'dataset': '/home/jpulsar/Dataset/COVID19/new_data',
                                            'prep': '/home/jpulsar/Dataset/COVID19/%s/org' % path_title['server'][0],
                                            'train': '/home/jpulsar/Dataset/COVID19/%s/prep' % path_title['server'][0]},
                             'output_path': {'log': '/home/jpulsar/Dataset/COVID19/log',
                                             'prep': '/home/jpulsar/Dataset/COVID19/%s' % path_title['server'][0],
                                             'train': '/home/jpulsar/Dataset/COVID19/%s' % path_title['server'][1]}}}


prep_param_set = {'npz_convert': {'class_type': '4cls',
                                  'data_limit': 50, # 각 클래스 별 사용할 데이터수, 전부 다 사용시 None
                                  'fig_mode': ['png'],
                                  'fig_sample': 0.2}, # 저장할 png 이미지 비율 (0~1)
                  'pre_process': {'data_mode': 'train',
                                  'data_limit': None,
                                  'step_list': [1,2,3,4,5],
                                  'Resize_param': {'size': [256, 256], 'mode': '2d'},
                                  'Adap_param': {'clip': 0.01},
                                  'fig_mode': ['png'],
                                  'fig_sample': 0.2}} # 저장할 png 이미지 비율 (0~1)

train_param_set = {'cls_vgg': {'train_mode': 'train',
                               'data_limit': None,
                               'model_name': 'VGG19', # VGG16, VGG19 중 선택
                               'step_list': [1,2,3,4,5],
                               'gpu_vision': '0', # nvida-smi에서 GPU 수 확인 (ex. '0,1,2')
                               'datagen': kimg.ImageDataGenerator(rotation_range=20,
                                                                  width_shift_range=0.1,
                                                                  height_shift_range=0.1,
                                                                  shear_range=0.1,
                                                                  zoom_range=0.1,
                                                                  horizontal_flip=True,
                                                                  vertical_flip=False,
                                                                  fill_mode='nearest'),
                               'Step1_param': {'class_mode': '3cls', # 3cls: [Covid19, Pneumonia, Normal], 4cls: [Covid19, Bacterial, Viral, Normal]
                                               'split_mode': 'num',
                                               'split_rate': {'test': 5, # class 당 test 데이터 숫자
                                                              'val': 10}}, # class 당 validation 데이터 숫자
                               'Step2_param': {'aug_rate': {'Covid19': 1, 'Bacterial': 1, 'Viral': 1, 'Pneumonia': 1, 'Normal': 1}, # class 당 augmentation 배율
                                               'fig_mode': 'png',
                                               'fig_sample': 0.2}, # 저장할 png 이미지 비율 (0~1)
                               'Step3_param': {'array_dim': '2d',
                                               'loss': losses.CategoricalCrossentropy(), # loss function
                                               'learning_rate': '1e-4', # learning rate (표기용)
                                               'optimizer': optimizers.Adam(lr=1e-4), # optimizer
                                               'metric': ['accuracy']},
                               'Step4_param': {'batch': 32, # Bacth
                                               'epoch': 50}, # Epoch 수
                               'Step5_param': {'pos_id1': {'name': 'Covid19',
                                                          'idx': 0},
                                               'pos_id2': {'name': 'Normal', # 'Bacterial', 'Viral', 'Pneumonia'
                                                           'idx': 1}},
                               'ensemble_mode': 'soft'}}

network_param_set = {'vgg': {'input_size': 256, # resize 이미지 크기와 일치
                             'block_num': 1, # conv block 수 조정
                             'layer_num': 1, # conv block 당 layer 수 조정
                             'drop_out': 0.5, # drop out 비율
                             'reg_val': '1e-4',
                             'conv_reg_weight': None,
                             'dens_reg_weight': regularizers.l1_l2(1e-4),
                             'dens_num': 2, # FC layer 수 조정
                             'dens_count': [1000, 500], # FC layer 당 filter 수
                             'output_count': 3, # 3cls, 4cls에 맞춰 일치
                             'conv_act': 'relu', # conv layer 활성화 함수 일괄 변경
                             'dens_act': 'relu', # FC layer 활성화 함수 일괄 변경
                             'output_act': 'softmax', # output layer 활성화 함수
                             'conv_str': 3, 
                             'pool_str': 2}}