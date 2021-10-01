from AI_COVID19.init import *
import AI_COVID19.ImageProcess.SubModules.DataIO as daio   # 데어터 불러오기 관련 모듈
import AI_COVID19.DeepLearningProcess.SubModules.TensorIO as tsio   # 텐서 처리 관련 모듈
import AI_COVID19.DeepLearningProcess.SubModules.ResultFunction as resf   # 결과 저정 관련 모듈
import AI_COVID19.DeepLearningProcess.NeuralNetwork.VGGModel as vggm   # VGG 신경망 모델


"""
신경망 학습을 진행하는 프로세스.
1. 데이터 불러오기, 2. 텐서 변환 3. 모델 준비, 4. 모델 학습, 5. 결과 저장의 5 단계로 이루어져있다.
데이터 확인 때는 1,2번 스텝만 사용, 테스트 때는 4번 스텝을 건너뛰면 된다.
"""

class TrainCLS:
    def __init__(self, train_params, network_params):   # Config 파일에서 파라미터를 입력받는 부분
        # Parameters dictionary
        self.train_param_dic = train_params   # 학습 조정과 관련된 파라미터들
        self.network_param_dic = network_params   # 신경망 구성과 관련된 파라미티들
        # Train Parameters
        self.train_mode = train_params['train_mode']   # Train of Test(미구현)
        self.data_limit = train_params['data_limit']   # 읽어들일 데이터 갯수, None 지정 시 전부 읽어들임.
        self.step_list = train_params['step_list']   # 사용할 스텝 ex) 데이터 구조 확인 1,2 번 스텝만 사용, 테스트시 1,2,3,5만 사용
        self.model_name = train_params['model_name']   # 사용할 신경망 모델의 이름
        self.gpu_vision = train_params['gpu_vision']   # 사용할 GPU 넘버
        self.datagen = train_params['datagen']   # Data Augmentation 파라미터
        self.Step1_param = train_params['Step1_param']   # 데이터 로딩 관련 파라미터
        self.Step2_param = train_params['Step2_param']   # 텐서 변환 관련 파라미터
        self.Step3_param = train_params['Step3_param']   # 모델 설계 관련 파라미터
        self.Step4_param = train_params['Step4_param']   # 모델 학습 관련 파라미터
        self.Step5_param = train_params['Step5_param']   # 결과 저장 관련 파라미터
        # Dictionary
        self.tensor_list = []

    def __call__(self, input_path, output_path, title, logger):
        daio.logging_title('Classification Training', logger, 'start')
        npz_path_list = daio.npz_path_reader(input_path, self.data_limit)
        num = len(npz_path_list) if len(npz_path_list) <= 100 else 100
        logger.info('Input File Path(n={}) = {}'.format(len(npz_path_list), npz_path_list[:num]))
        logger.info('(* Showing Patient List <= 100)')
        logger.info('')
        # output path build
        save_path = os.path.join(output_path, title)
        daio.path_builder(save_path)
        # GPU Setting
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_vision  # '0,1,2,3'
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            # Step 1. Dataset Setting
            if self.step_list.count(1):
                logger.info('')
                logger.info('### 1. Dataset Setting ###')
                logger.info('')

                # Make Tensor list
                for i, i_ in enumerate(npz_path_list):
                    data_dic = daio.npz_file_reader(i_, 'npz')
                    self.tensor_list.append(data_dic)

                # Class Split
                cls_tensor_list = tsio.class_split(self.tensor_list, self.Step1_param['class_mode'])
                logger.info('class category = %s' % list(cls_tensor_list.keys()))
                cls_tensor_listnum = []
                for i, i_ in enumerate(list(cls_tensor_list.keys())):
                    cls_tensor_listnum.append(len(cls_tensor_list[i_]))
                logger.info('class number = %s' % cls_tensor_listnum)

                # Train-Val-Test split
                clstvt_tensor_list = tsio.train_test_split(cls_tensor_list, self.Step1_param['split_rate'], self.Step1_param['split_mode'])
                clstvt_tensor_listnum = {'all': {'Train': 0, 'Val': 0, 'Test': 0}}
                for i, i_ in enumerate(list(clstvt_tensor_list.keys())):
                    clstvt_tensor_listnum.update({i_: {'Train': [], 'Val': [], 'Test': []}})
                    for j, j_ in enumerate(list(clstvt_tensor_list[i_].keys())):
                        clstvt_tensor_listnum[i_][j_] = len(clstvt_tensor_list[i_][j_])
                        clstvt_tensor_listnum['all'][j_] += len(clstvt_tensor_list[i_][j_])
                    logger.info('\'{}\' Class: Train-Val-Test Split Result = {}'.format(i_, clstvt_tensor_listnum[i_]))
                logger.info('\"All\" Class: Train-Val-Test Split Result = {}'.format(clstvt_tensor_listnum['all']))

            # Step 2. Tensor Setting
            if self.step_list.count(2):
                logger.info('')
                logger.info('### 2. Tensor Setting ###')
                logger.info('')
                if self.Step1_param['class_mode'] == '4cls':
                    # Tensor Convering
                    cov_tensor_dic, cov_name_dic = self.tensor_coverting(clstvt_tensor_list['Covid19'], 'Covid19',
                                                                         self.datagen, save_path, logger)
                    bac_tensor_dic, bac_name_dic = self.tensor_coverting(clstvt_tensor_list['Bacterial'], 'Bacterial',
                                                                         self.datagen, save_path, logger)
                    vir_tensor_dic, vir_name_dic = self.tensor_coverting(clstvt_tensor_list['Viral'], 'Viral',
                                                                         self.datagen, save_path, logger)
                    nor_tensor_dic, nor_name_dic = self.tensor_coverting(clstvt_tensor_list['Normal'], 'Normal',
                                                                         self.datagen, save_path, logger)

                    # Class Merging
                    train_x = np.concatenate((cov_tensor_dic['Train'][0], bac_tensor_dic['Train'][0], vir_tensor_dic['Train'][0], nor_tensor_dic['Train'][0]), axis=0)
                    train_y = np.concatenate((cov_tensor_dic['Train'][1], bac_tensor_dic['Train'][1], vir_tensor_dic['Train'][1], nor_tensor_dic['Train'][1]), axis=0)
                    val_x = np.concatenate((cov_tensor_dic['Val'][0], bac_tensor_dic['Val'][0], vir_tensor_dic['Val'][0], nor_tensor_dic['Val'][0]), axis=0)
                    val_y = np.concatenate((cov_tensor_dic['Val'][1], bac_tensor_dic['Val'][1], vir_tensor_dic['Val'][1], nor_tensor_dic['Val'][1]), axis=0)
                    test_x = np.concatenate((cov_tensor_dic['Test'][0], bac_tensor_dic['Test'][0], vir_tensor_dic['Test'][0], nor_tensor_dic['Test'][0]), axis=0)
                    test_y = np.concatenate((cov_tensor_dic['Test'][1], bac_tensor_dic['Test'][1], vir_tensor_dic['Test'][1], nor_tensor_dic['Test'][1]), axis=0)
                    logger.info('')
                    logger.info('All Train dataset shape = {}, {}'.format(train_x.shape, train_y.shape))
                    logger.info('All Validation dataset shape = {}, {}'.format(val_x.shape, val_y.shape))
                    logger.info('All Test dataset shape = {}, {}'.format(test_x.shape, test_y.shape))
                    train_name_list = cov_name_dic['Train'] + bac_name_dic['Train'] + vir_name_dic['Train'] + nor_name_dic['Train']
                    val_name_list = cov_name_dic['Val'] + bac_name_dic['Val'] + vir_name_dic['Val'] + nor_name_dic['Val']
                    test_name_list = cov_name_dic['Test'] + bac_name_dic['Test'] + vir_name_dic['Test'] + nor_name_dic['Test']
                elif self.Step1_param['class_mode'] == '3cls':
                    # Tensor Convering
                    cov_tensor_dic, cov_name_dic = self.tensor_coverting(clstvt_tensor_list['Covid19'], 'Covid19', self.datagen, save_path, logger)
                    pne_tensor_dic, pne_name_dic = self.tensor_coverting(clstvt_tensor_list['Pneumonia'], 'Pneumonia', self.datagen, save_path, logger)
                    nor_tensor_dic, nor_name_dic = self.tensor_coverting(clstvt_tensor_list['Normal'], 'Normal', self.datagen, save_path, logger)

                    # Class Merging
                    train_x = np.concatenate((cov_tensor_dic['Train'][0], pne_tensor_dic['Train'][0], nor_tensor_dic['Train'][0]), axis=0)
                    train_y = np.concatenate((cov_tensor_dic['Train'][1], pne_tensor_dic['Train'][1], nor_tensor_dic['Train'][1]), axis=0)
                    val_x = np.concatenate((cov_tensor_dic['Val'][0], pne_tensor_dic['Val'][0], nor_tensor_dic['Val'][0]), axis=0)
                    val_y = np.concatenate((cov_tensor_dic['Val'][1], pne_tensor_dic['Val'][1], nor_tensor_dic['Val'][1]), axis=0)
                    test_x = np.concatenate((cov_tensor_dic['Test'][0], pne_tensor_dic['Test'][0], nor_tensor_dic['Test'][0]), axis=0)
                    test_y = np.concatenate((cov_tensor_dic['Test'][1], pne_tensor_dic['Test'][1], nor_tensor_dic['Test'][1]), axis=0)
                    logger.info('')
                    logger.info('All Train dataset shape = {}, {}'.format(train_x.shape, train_y.shape))
                    logger.info('All Validation dataset shape = {}, {}'.format(val_x.shape, val_y.shape))
                    logger.info('All Test dataset shape = {}, {}'.format(test_x.shape, test_y.shape))
                    train_name_list = cov_name_dic['Train'] + pne_name_dic['Train'] + nor_name_dic['Train']
                    val_name_list = cov_name_dic['Val'] + pne_name_dic['Val'] + nor_name_dic['Val']
                    test_name_list = cov_name_dic['Test'] + pne_name_dic['Test'] + nor_name_dic['Test']

                # Patient List Save
                df_patient_list = pd.DataFrame({'train': train_name_list,
                                                'val': val_name_list + [None] *
                                                       (len(train_name_list) - len(val_name_list)),
                                                'test': test_name_list + [None] * (
                                                        len(train_name_list) - len(test_name_list))})
                df_patient_list.to_csv(os.path.join(save_path, self.model_name + '_patient_list.csv'))

            # Step 3. Network Model Build
            if self.step_list.count(3):
                logger.info('')
                logger.info('### 3. Model Build ###')
                logger.info('')
                model = self.network_model_build(self.model_name, self.Step3_param['array_dim'], self.network_param_dic)
                model.summary(print_fn=logger.info)
                logger.info('### {} Model Build! ###'.format(self.model_name))
                model.compile(loss=self.Step3_param['loss'], optimizer=self.Step3_param['optimizer'], metrics=self.Step3_param['metric'])

            # Step 4. Model Training
            if self.step_list.count(4):
                logger.info('')
                logger.info('### 4. Model Training ###')
                logger.info('')
                log_save_path = os.path.join(save_path, 'log_{}'.format(self.train_mode))
                daio.path_builder(log_save_path)
                callback_list = [keras.callbacks.EarlyStopping(monitor='val_' + self.Step3_param['metric'][0], patience=20, mode='max'),
                                 keras.callbacks.ModelCheckpoint(
                                     filepath=os.path.join(save_path, title + '_model.h5'),
                                     monitor='val_' + self.Step3_param['metric'][0], save_best_only=True, mode='max'),
                                 keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='min')]
                # keras.callbacks.TensorBoard(log_dir=log_save_path, histogram_freq=1, embeddings_freq=1)

                history = model.fit(train_x, train_y, batch_size=self.Step4_param['batch'],
                                    epochs=self.Step4_param['epoch'], validation_data=(val_x, val_y),
                                    callbacks=callback_list, shuffle=True)

                # History Save
                if self.step_list.count(3):
                    df_history = pd.DataFrame(history.history)
                    logger.info('<Train History>')
                    logger.info(df_history)
                    df_history.to_csv(os.path.join(save_path, self.model_name + '_history.csv'))
                    resf.loss_acc_graph(save_path, self.model_name, self.Step3_param['metric'])
                    logger.info('# Network Model {} Train is Finished!'.format(self.model_name))
                    logger.info('# Result Save Path = {}'.format(save_path))

            # Step 5. Model Evaluate
            if self.step_list.count(5):
                logger.info('')
                logger.info('### 5. Model Evaluate ###')
                logger.info('')
                # Model Evaluate
                model.load_weights(os.path.join(save_path, title + '_model.h5'))
                eval_results = resf.train_evaluator(model, {'val': val_x, 'test': test_x},
                                                    {'val': val_y, 'test': test_y},
                                                    self.Step4_param['batch'], self.train_param_dic, self.network_param_dic,
                                                    save_path, self.model_name)
                logger.info(
                    'Validation Result: Loss = {}, Accuracy = {}'.format(eval_results['val'][0], eval_results['val'][1]))
                logger.info(
                    'Test Result: Loss = {}, Accuracy = {}'.format(eval_results['test'][0], eval_results['test'][1]))

                # Model Prediction
                val_results = resf.train_predictor(model, val_name_list, val_x, val_y, 'val', self.Step4_param['batch'],
                                                   save_path, self.model_name, self.Step1_param['class_mode'])
                test_results = resf.train_predictor(model, test_name_list, test_x, test_y, 'test', self.Step4_param['batch'],
                                                    save_path, self.model_name, self.Step1_param['class_mode'])
                logger.info('Validation label shape = {}, Validation prediction shape = {}'.format(val_y.shape,
                                                                                             val_results.shape))
                logger.info('Test label shape = {}, Test prediction shape = {}'.format(test_y.shape, test_results.shape))

                # Prediction Result Save
                pred_path = os.path.join(save_path, 'pred')
                daio.path_builder(pred_path)
                val_tru = resf.onehot_class_converter(val_y, self.Step1_param['class_mode'])
                val_pre = resf.onehot_class_converter(val_results, self.Step1_param['class_mode'])
                for i in range(len(val_name_list)):
                    daio.npz_file_writer(pred_path, title + '_val', 'pred', {'name': val_name_list[i], 'image': val_x[i],
                                                                             'label': val_tru[i], 'pred': val_pre[i]})
                test_tru = resf.onehot_class_converter(test_y, self.Step1_param['class_mode'])
                test_pre = resf.onehot_class_converter(test_results, self.Step1_param['class_mode'])
                for i in range(len(test_name_list)):
                    daio.npz_file_writer(pred_path, title + '_test', 'pred',
                                         {'name': test_name_list[i], 'image': test_x[i],
                                          'label': test_tru[i], 'pred': test_pre[i]})

                # Confusion Matrix
                val_clsrep, val_confmat = resf.result_confusion_matrix(val_tru, val_pre, save_path, title + '_val', self.Step1_param['class_mode'])
                logger.info('\n# Validation Classification Report\n')
                logger.info(val_clsrep)
                logger.info('# Validation Confusion Matrix\n')
                logger.info(val_confmat)
                test_clsrep, test_confmat = resf.result_confusion_matrix(test_tru, test_pre, save_path, title + '_test', self.Step1_param['class_mode'])
                logger.info('\n# Test Classification Report\n')
                logger.info(test_clsrep)
                logger.info('# Test Confusion Matrix\n')
                logger.info(test_confmat)

                # ROC_AUC(COVID19)
                val_roc = resf.result_roc_auc(val_y, val_results, self.Step5_param['pos_id1'], save_path, title + '_val')
                test_roc = resf.result_roc_auc(test_y, test_results, self.Step5_param['pos_id1'], save_path, title + '_test')
                print('ROC_AUC Value(Covid19): Validation = {}, Test = {}'.format(val_roc, test_roc))

                # ROC_AUC(Pneumonia)
                val_roc = resf.result_roc_auc(val_y, val_results, self.Step5_param['pos_id2'], save_path, title + '_val')
                test_roc = resf.result_roc_auc(test_y, test_results, self.Step5_param['pos_id2'], save_path, title + '_test')
                print('ROC_AUC Value(Pneumonia): Validation = {}, Test = {}'.format(val_roc, test_roc))

    def tensor_coverting(self, tensor_list, class_name, data_generator, path, logger):
        train_x, train_y = tsio.tensor_setting(tensor_list['Train'])
        val_x, val_y = tsio.tensor_setting(tensor_list['Val'])
        test_x, test_y = tsio.tensor_setting(tensor_list['Test'])
        logger.info('({}) Train dataset shape = {}, {}'.format(class_name, train_x.shape, train_y.shape))
        logger.info('({}) Validation shape = {}, {}'.format(class_name, val_x.shape, val_y.shape))
        logger.info('({}) Test dataset shape = {}, {}'.format(class_name, test_x.shape, test_y.shape))

        data_generator.fit(train_x)
        aug_train_x, aug_train_y, num = [], [], 0
        for batch in data_generator.flow(train_x, y=train_y, batch_size=1, shuffle=False):
            if num >= (self.Step2_param['aug_rate'][class_name] - 1) * train_x.shape[0]:
                break
            aug_train_x.append(batch[0][0])
            aug_train_y.append(batch[1][0])
            num += 1

        # All Train Dataset
        if len(aug_train_x) == 0:
            all_train_x, all_train_y = train_x, train_y
        else:
            all_train_x = np.concatenate((train_x, np.array(aug_train_x, dtype=np.uint8)), axis=0)
            all_train_y = np.concatenate((train_y, np.array(aug_train_y, dtype=np.uint8)), axis=0)
        logger.info('({}) Aug_Train dataset shape = {}, {}'.format(class_name, all_train_x.shape, all_train_y.shape))
        logger.info('')
        tensor_dic = {'Train': [all_train_x, all_train_y], 'Val': [val_x, val_y], 'Test': [test_x, test_y]}

        # All Tensor figure Save
        train_name_list = self.tensor_saver(train_x, tensor_list['Train'], 'train', self.Step2_param, path)
        aug_train_name_list = self.tensor_saver(aug_train_x, tensor_list['Train'], 'train_aug', self.Step2_param,
                                                path)
        val_name_list = self.tensor_saver(val_x, tensor_list['Val'], 'val', self.Step2_param, path)
        test_name_list = self.tensor_saver(test_x, tensor_list['Test'], 'test', self.Step2_param, path)
        name_dic = {'Train': train_name_list, 'Val': val_name_list, 'Test': test_name_list}
        return tensor_dic, name_dic

    def tensor_saver(self, tensor, tensor_list, title, param_dic, output_path):
        tensor_name_list = []
        for i, i_ in enumerate(tensor):
            if i >= len(tensor_list):
                break
            name = tensor_list[i]['name']
            tensor_name_list.append(name)
            if param_dic['fig_sample'] != 0:
                if i % int(1 / param_dic['fig_sample']) == 0:
                    daio.fig_saver(output_path, title, param_dic['fig_mode'], name, i_[..., 0])
        return tensor_name_list

    def network_model_build(self, model_name, array_dim, network_params):
        if array_dim == '2d':
            if model_name[:3] == 'VGG':
                model = vggm.VGG(network_params)(model_name, array_dim)
            elif model_name[:3] == 'Res':
                model = resm.ResNet(network_params)(model_name, array_dim)
            else:
                print('\n !ERROR! Please Insert Correct Model Name (VGG, ResNet, etc..) \n')
        return model





