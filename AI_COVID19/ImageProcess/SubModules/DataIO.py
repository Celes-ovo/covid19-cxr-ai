from AI_COVID19.init import *


# Path Modules
def path_builder(path):
    """
    원하는 경로에 폴더를 생성하는 함수, 이미 존재할시 넘어간다.
    :param path: 만들고 싶은 폴더의 경로
    :return: 해당 경로에 폴더 생성
    """
    try:
        os.mkdir(path)
    except:
        pass


def file_path_reader(input_path, data_limit):
    """
    해당 경로 안의 파일들의 경로를 읽어 리스트화 시켜주는 모듈
    :param input_path: 데이터셋이 존재하는 폴더의 경로
    :param data_limit: 읽어햘 데이터셋의 갯수, None으로 지정시 전체 읽음
    :return: 폴더 안의 파일들의 경로를 모두 읽어 리스트
    """
    file_path_list = []
    for i, i_ in enumerate(sorted(os.listdir(input_path))[:data_limit]):
        file_path_list.append(os.path.join(input_path, i_))
    return file_path_list


def npz_path_reader(input_path, data_limit, remove_list=[]):
    """
    경로 안의 NPZ 파일들을 읽는 모듈
    :param input_path: NPZ 파일들이 모여있는 폴더의 경로
    :param data_limit: 읽어햘 NPZ 파일의 갯수, None으로 지정시 전체 읽음
    :param remove_list: 특별히 제거하고 싶은 NPZ 파일 목록
    :return: NPZ 파일 경로 리스트
    """
    npz_path = []
    for i, i_ in enumerate(sorted(os.listdir(input_path))[:data_limit]):
        if remove_list.count(i_) == 0:
            npz_path.append(os.path.join(input_path, i_))
    return npz_path


# Header Modules
def header_writer(sitk_image, header_dic):
    """
    SimpleITK Object 파일에 헤더를 저장시킴
    :param sitk_image: 저장시켜야할 SimpleITK Object
    :param header_dic: 저장시키고 싶은 헤더들의 딕셔너리
    :return:
    """
    sitk_image.SetSpacing(header_dic['spacing'])
    sitk_image.SetOrigin(header_dic['origin'])
    sitk_image.SetDirection(header_dic['direction'])


def header_extracter(sitk_image):
    """
    SimpleITK Object에서 헤더를 추출하여 딕셔너리로 만듬
    :param sitk_image: 헤더를 추출할 SimpleITK Object
    :return: 추출된 헤더의 딕셔너리
    """
    return {'size': sitk_image.GetSize(),
            'spacing': sitk_image.GetSpacing(),
            'origin': sitk_image.GetOrigin(),
            'dimension': sitk_image.GetDimension(),
            'direction': sitk_image.GetDirection(),
            'pixel_id': sitk_image.GetPixelID(),
            'pixel_components_number': sitk_image.GetNumberOfComponentsPerPixel()}


# NPZ Module
def npz_file_reader(input_path, mode):
    """
    저장된 NPZ 파일을 읽는 모듈
    :param input_path: NPZ 파일의 경로
    :param mode: 'npz' or 'tensor'
    :return:
    """
    data_dic = {} # Single Dataset Dictionary
    if mode == 'npz':
        name = np.load(input_path)['name']
        header = np.load(input_path, allow_pickle=True)['header'][()]
        image = np.load(input_path, allow_pickle=True)['image']
        info = np.load(input_path, allow_pickle=True)['info'][()]
        data_dic.update({'name': str(name), 'header': header, 'image': image,
                         'info': info})
        return data_dic
    elif mode == 'tensor':
        name = np.load(input_path)['name']
        image = np.load(input_path, allow_pickle=True)['image']
        label = np.load(input_path, allow_pickle=True)['label']
        data_dic.update({'name': name, 'image': image, 'label': label})
        return data_dic
    else:
        print('please typed correctly mode!')


def npz_file_writer(output_path, folder, mode, dataset_dict):
    """
    NPZ 파일로 저장하는 모듈
    :param output_path: 저장 경로
    :param folder: 저장시킬 폴더의 이름
    :param mode: 'npz' or 'tensor'
    :param dataset_dict: 저장 시켜야할 딕셔너리 형태의 데이터셋
    :return:
    """
    try:
        os.mkdir(os.path.join(output_path, folder))
    except:
        pass
    output_file_path = os.path.join(output_path, folder)
    if mode == 'npz':
        np.savez(os.path.join(output_file_path, '{}.npz'.format(dataset_dict['name'])),
                 name=dataset_dict['name'],
                 header=dataset_dict['header'],
                 image=dataset_dict['image'],
                 info=dataset_dict['info'])
    elif mode == 'tensor':
        np.savez(os.path.join(output_file_path, '{}.npz'.format(dataset_dict['name'])),
                 name=dataset_dict['name'],
                 header=dataset_dict['header'],
                 image=dataset_dict['image'],
                 aug_img = dataset_dict['aug_img'],
                 label=dataset_dict['info'])
    elif mode == 'pred':
        np.savez(os.path.join(output_file_path, '{}.npz'.format(dataset_dict['name'])),
                 name=dataset_dict['name'],
                 image=dataset_dict['image'],
                 label=dataset_dict['label'],
                 pred=dataset_dict['pred'])
    else:
        print('please typed correctly mode!')


# Save Module
def logging_title(text, logger, mode):
    """
    각 프로세스의 시작과 끝을 알려주는 Log 모듈
    :param text: 출력하고 싶은 메인 단어
    :param logger: logger
    :param mode: 'start' or 'end'
    :return:
    """
    if mode == 'start':
        logger.info('')
        logger.info('=' * 100)
        logger.info('### !{} START! ###'.format(text))
        logger.info('-' * 100)
        logger.info('')
    elif mode == 'end':
        logger.info('')
        logger.info('-' * 100)
        logger.info('### !{} COMPLETE! ###'.format(text))
        logger.info('=' * 100)
        logger.info('')
    else:
        logger.error('Please Insert Correct mode(\'start\' or \'end\')')


def fig_saver(output_path, title, mode, name, array):
    """
    시각화를 위하여 각가의 영상을 이미지 파일로 저장하는 모듈
    :param output_path: 저장 경로
    :param title: 데이터셋 이름
    :param mode: 'png' or 'dcm'
    :param name: 파일 이름
    :param array: 저장할 이미지(format: Numpy Array)
    :return:
    """
    path_builder(os.path.join(output_path, 'figure'))
    fig_save_path = os.path.join(output_path, 'figure', title + '_' + mode)
    path_builder(fig_save_path)
    if mode == 'png':
        if len(array.shape) == 3:
            if array.shape[-1] == 4:
                mimg.imsave(os.path.join(fig_save_path, '{}.png'.format(name)), array[...,0], cmap='gray')
            else:
                mimg.imsave(os.path.join(fig_save_path, '{}.png'.format(name)), array, cmap='gray')
        else:
            mimg.imsave(os.path.join(fig_save_path, '{}.png'.format(name)), array, cmap='gray')
    elif mode == 'dcm':
        sitk_image = sitk.GetImageFromArray(np.array(array, dtype=np.uint16), isVector=False)
        sitk_image.SetMetaData('0010|0020', str(name)), sitk_image.SetMetaData('0008|1030', str(name))
        sitk.WriteImage(sitk_image, os.path.join(fig_save_path, '{}.dcm'.format(name)))
    else:
        print('!ERROR! please typed correctly fig_mode!')


def config_saver(name, param_dict, patients, output_path, title):
    """
    전처리 또는 각종 프로세스에 사용된 환자명을 저장하고, 사용한 파라미터를 정하는 모듈
    :param name: 데이터셋 또는 프로세스 이름
    :param param_dict: 사용한 파라미터 딕셔너리
    :param patients: 환자 리스트
    :param output_path: 저장경로
    :param title: 저장할 이름름
   :return:
    """
    num_sub = len(patients['npz_name']) - len(patients['err_name'])
    if num_sub >= 0:
        df_patient_list = pd.DataFrame({'patient_list({})'.format(name): patients['npz_name'],
                                        'original_list': patients['org_name'],
                                        'error_list': patients['err_name'] + [''] * abs(num_sub)})
    else:
        df_patient_list = pd.DataFrame({'patient_list({})'.format(name): patients['npz_name'] + [''] * abs(num_sub),
                                        'original_list': patients['org_name'] + [''] * abs(num_sub),
                                        'error_list': patients['err_name']})
    df_patient_list.to_csv(os.path.join(output_path, title + '_patient_list.csv'))
    config = {}
    config.update({'# {} #'.format(name): None})
    config.update(param_dict)
    df_config = pd.DataFrame({'keys': list(config.keys()), 'values': list(config.values())})
    df_config.to_csv(os.path.join(output_path, title + '_config.csv'))


