from AI_COVID19.init import *   # 모든 파이썬 라이브러리 임포트
from AI_COVID19.ImageProcess.SubModules import DataIO as daio   # 데이터 IO 관련 모듈
from AI_COVID19.ImageProcess.SubModules import PreProcess as prep    # 전처리 관련 모듈


"""
이미지 전처리를 수행하는 클래스
Color to grey, Zero Padding, Resize, CLAHE, Min-Max Normalization 등으로 구성되어있으며,
각각의 전처리 과정을 빼고 더할 수 있다.
"""

class PreProcessing:
    def __init__(self, prep_params):   # Config 파일에서 파라미터 입력부
        # parameter dictionary
        self.param_dic = prep_params
        # Pre-processing parameters
        self.data_mode = prep_params['data_mode']   # Train of Test(미구현)
        self.data_limit = prep_params['data_limit']   # 읽어들일 데이터 갯수, None으로 지정시 모든 데이터셋 불러옴
        self.step_list = prep_params['step_list']   # 전처리 스텝 관리(필요한 스텝만 사용가능)
        self.Resize_param = prep_params['Resize_param']   # Resize 관련 파라미터
        self.Adap_param = prep_params['Adap_param']   # CLAHE 관련 파라미터
        self.fig_mode = prep_params['fig_mode']  # 영상 결과만 보기 위해, 'png' 또는 'dcm' 등으로 저장
        self.fig_sample = prep_params['fig_sample']   # 전체 데이터셋 중 이미지로 저장할 비율 0~1 사이 지정
        # patient List
        self.patient_list = {'org_name': [], 'npz_name': [], 'err_name': []}

    def __call__(self, input_path, output_path, title, logger):
        daio.logging_title('Image Pre-Processing', logger, 'start')
        npz_path_list = daio.npz_path_reader(input_path, self.data_limit)
        logger.info('Input File Path(n={}) = {}'.format(len(npz_path_list), npz_path_list))
        logger.info('')

        # Pre-processing
        for i, i_ in enumerate(npz_path_list):
            try:
                logger.info('# Image_({}/{}). Image Processing file = {}'.format(i + 1, len(npz_path_list), i_))
                data_dic = daio.npz_file_reader(i_, 'npz')
                data_dic['info'].update({'prep_log': []})
                logger.info(' - 0. Origin Image: Name = {}, Header = {}'.format(data_dic['name'], data_dic['header']))
                # Step 1. Color to grey
                if self.step_list.count(1):
                    data_dic['image'], data_dic['header'] = prep.make_rgb_grey(data_dic['image'], data_dic['header'])
                    if data_dic['header']['pixel_components_number'] == 1:
                        data_dic['info']['channel'] = 'grey'
                    data_dic['info']['prep_log'].append('step1_Color_to_grey')
                    logger.info(' - 1. Color to grey Result: Pixel Components Number = {}, image channel = {}'
                                .format(data_dic['header']['pixel_components_number'], data_dic['info']['channel']))
                # Step 2. Zero padding
                if self.step_list.count(2):
                    data_dic['image'], data_dic['header'] = prep.zero_padding(data_dic['image'], data_dic['header'])
                    data_dic['info']['prep_log'].append('step2_Zero_Padding')
                    logger.info(' - 2. Zero Padding Image Result: Image size = {}'.format(data_dic['header']['size']))
                # Step 3. Resizing
                if self.step_list.count(3):
                    data_dic['image'], data_dic['header'] = prep.resize_array(data_dic['image'], data_dic['header'], self.Resize_param['size'], self.Resize_param['mode'])
                    data_dic['info']['prep_log'].append('step3_Resizing')
                    logger.info(' - 3. Resizing Image Header = {}'.format(data_dic['header']))
                # Step 4. Adaptive Equalization
                if self.step_list.count(4):
                    data_dic['image'] = prep.adapequal_array(data_dic['image'], self.Adap_param['clip'])
                    data_dic['info']['prep_log'].append('step4_Adaptive_Equalization')
                    logger.info(' - 4. Adaptive Equalization Result: size = {}, min = {}, max = {}, mean = {}, std = {}'
                                .format(data_dic['image'].shape, np.min(data_dic['image']), np.max(data_dic['image']),
                                        np.mean(data_dic['image']), np.std(data_dic['image'])))
                # Step 5. MinMax_Normalization
                if self.step_list.count(5):
                    data_dic['image'] = prep.norm_array(data_dic['image'])
                    data_dic['info']['prep_log'].append('step5_MinMax_Normalization')
                    logger.info(' - 5. Min-Max Normalization Result: size = {}, min = {}, max = {}, mean = {}, std = {}'
                                .format(data_dic['image'].shape, np.min(data_dic['image']), np.max(data_dic['image']),
                                        np.mean(data_dic['image']), np.std(data_dic['image'])))
                # Save dataset
                self.patient_list['npz_name'].append(data_dic['name'])
                self.patient_list['org_name'].append(i_)
                logger.info(' - #. Preprocess Completed Image Info. = {}'.format(data_dic['info']))
                if self.data_mode == 'train':
                    daio.npz_file_writer(output_path, title, 'npz', data_dic)
                if self.data_mode == 'test':
                    daio.npz_file_writer(output_path, title, 'tensor', data_dic)
                if self.fig_sample != 0:
                    if i % int(1 / self.fig_sample) == 0:
                        if self.fig_mode.count('png'):
                            daio.fig_saver(output_path, title, 'png', data_dic['name'], data_dic['image'])
                        if self.fig_mode.count('dcm'):
                            daio.fig_saver(output_path, title, 'dcm', data_dic['name'], data_dic['image'])
            except Exception as err:
                self.patient_list['err_name'].append(i_)
                logger.error('!ERROR! Step_({}/{}). Error file = {}'.format(i + 1, len(npz_path_list), i_))
                logger.error(err)
                logger.info('')
        # Result Log
        logger.info('# Image Pre-Processing Result #')
        logger.info('Output File Path = {}'.format(os.path.join(output_path, title, '*.npz')))
        logger.info('Total Patient number = {} & Error Number = {}'.format(len(self.patient_list['npz_name']),
                                                                           len(self.patient_list['err_name'])))
        num = len(self.patient_list['npz_name']) if len(self.patient_list['npz_name']) <= 100 else 100
        logger.info('Patient List = {}'.format(self.patient_list['npz_name'][:num]))
        logger.info('(* Showing Patient List <= 100')
        logger.info('Error List = {}'.format(self.patient_list['err_name']))
        daio.config_saver('Image Pre-Processing', self.param_dic, self.patient_list, output_path, title)
        daio.logging_title('Image Pre-Processing', logger, 'end')

