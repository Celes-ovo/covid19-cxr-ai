from AI_COVID19.init import *   # 모든 파이썬 라이브러리 임포트
from AI_COVID19.ImageProcess.SubModules import DataIO as daio   # Data IO 관련 모듈

"""
원본 데이터셋을 찾고, 읽은 후 dictionary 형태의 NPZ 파일로 모든 데이터셋을 변환하는 클래스
dataset = {'name': 이름,
           'header': 이미지 헤더 정보(if Dicom일 경우, 현재 대부분 이미지가 PNG라 의미는 없음...),
           'image': 영상 Numpy Array,
           'info': {'cls_label': 클래스 종류(COVID19, Bacterial, Viral, Normal 등등),
                    'dataset': 데이터셋 출처(보라매, Covidx, Kaggle 등등),
                    'channel': color or grey}} 
"""
class NpzConverter:
    def __init__(self, prep_params):  # Config 코드로부터 파라미터 입력부
        # parameter dictionary
        self.param_dic = prep_params
        # Data IO parameters
        self.class_type = prep_params['class_type']   # 4cls or 3cls(미구현)
        self.data_limit = prep_params['data_limit']   # 각 데이터셋 폴더에서 읽어들일 데이터 갯수, None으로 지정시 모든 데이터셋 불러옴
        self.fig_mode = prep_params['fig_mode']    # 영상 결과만 보기 위해, 'png' 또는 'dcm' 등으로 저장
        self.fig_sample = prep_params['fig_sample']    # 전체 데이터셋 중 이미지로 저장할 비율 0~1 사이 지정
        # Data IO List
        self.patient_list = {'org_name': [], 'npz_name': [], 'err_name': []}  # org_name: 원본 파일명, npz_name: npz 저장명, err_name: 에러 파일명


    def __call__(self, input_path, output_path, title, logger):
        daio.logging_title('COVID19 Data NPZ Converting', logger, 'start')
        # All Dataset Loading
        logger.info('Dataset Category = %s' % os.listdir(input_path))
        if self.class_type == '4cls':
            cvd_path = daio.file_path_reader(os.path.join(input_path, 'covid'), self.data_limit)
            bac_path = daio.file_path_reader(os.path.join(input_path, 'bacteria'), self.data_limit)
            vir_path = daio.file_path_reader(os.path.join(input_path, 'virus'), self.data_limit)
            nor_path = daio.file_path_reader(os.path.join(input_path, 'covid'), self.data_limit)
        logger.info(' - COVID Patient Number = %s' % (len(cvd_path)))
        logger.info(' - Bacterial Pneumonia Patient Number = %s' % len(bac_path))
        logger.info(' - Viral Pneumonia Patient Number = %s' % len(vir_path))
        logger.info(' - Normal Patient Number = %s' % len(nor_path))

        # All Dataset data Build
        if self.class_type == '4cls':
            self.npz_builder(cvd_path, 'CovidX', 'Covid19', output_path, title, logger, self.patient_list)
            self.npz_builder(bac_path, 'Kaggle', 'Bacterial', output_path, title, logger, self.patient_list)
            self.npz_builder(vir_path, 'Kaggle', 'Viral', output_path, title, logger, self.patient_list)
            self.npz_builder(nor_path, 'CovidX', 'Normal', output_path, title, logger, self.patient_list)


        # Result Log
        logger.info('# Data NPZ Converting Result #')
        logger.info('Output File Path = {}'.format(os.path.join(output_path, title, '*.npz')))
        logger.info('Total Patient number = {} & Error Number = {}'.format(len(self.patient_list['npz_name']), len(self.patient_list['err_name'])))
        num = len(self.patient_list['npz_name']) if len(self.patient_list['npz_name']) <= 100 else 100
        logger.info('Patient List = {}'.format(self.patient_list['npz_name'][:num]))
        logger.info('(* Showing Patient List <= 100')
        logger.info('Error List = {}'.format(self.patient_list['err_name']))
        daio.config_saver('COVID19 Data NPZ Converting', self.param_dic, self.patient_list, output_path, title)
        daio.logging_title('COVID19 Data NPZ Converting', logger, 'end')

    def npz_builder(self, path_list, name, label, output_path, title, logger, patient_list):
        logger.info('# \'{}\' Dataset, \'{}\' Class Conversion!'.format(name, label))
        for i, i_ in enumerate(path_list):
            try:
                data_dic = {}
                dataset_sitk = sitk.ReadImage(i_)
                data_dic['name'] = label + '_' + name + '_n' + str(10001 + i)[1:]
                data_dic['header'] = daio.header_extracter(dataset_sitk)
                if data_dic['header']['pixel_components_number'] == 1:
                    data_chn = 'grey'
                else:
                    data_chn = 'color'
                data_dic['image'] = sitk.GetArrayFromImage(dataset_sitk)
                data_dic['info'] = {'cls_label': label, 'dataset': name, 'channel': data_chn}
                # log
                logger.info('Step_({}/{}). Completed file = {}'.format(i + 1, len(path_list), i_))
                logger.info('   patient = {}, image shape = {}, info = {}'.format(data_dic['name'], data_dic['image'].shape,
                                                                                  data_dic['info']))
                logger.info('   header = {}'.format(data_dic['header']))
                logger.info('')
                # Save dataset
                patient_list['npz_name'].append(data_dic['name'])
                patient_list['org_name'].append(i_)
                daio.npz_file_writer(output_path, title, 'npz', data_dic)
                if self.fig_sample != 0:
                    if i % int(1 / self.fig_sample) == 0:
                        if self.fig_mode.count('png'):
                            daio.fig_saver(output_path, title, 'png', data_dic['name'], data_dic['image'])
                        if self.fig_mode.count('dcm'):
                            daio.fig_saver(output_path, title, 'dcm', data_dic['name'], data_dic['image'])
            except Exception as err:
                patient_list['err_name'].append(i_)
                logger.error('!ERROR! Step_({}/{}). Error file = {}'.format(i + 1, len(path_list), i_))
                logger.error(err)
                logger.info('')


