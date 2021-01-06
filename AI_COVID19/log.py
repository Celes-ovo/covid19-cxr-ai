from AI_COVID19.init import *
import logging
import AI_COVID19.ImageProcess.SubModules.DataIO as daio

"""
로그를 세팅하고, 결과 폴더를 만드는 모듈
"""


def logger_setting(path, data_mode):
    if data_mode == 'train':
        daio.path_builder(path['log'])
        daio.path_builder(path['prep'])
        daio.path_builder(path['train'])
        log_name = 'log_train_{}_{}.txt'.format(os.path.basename(path['prep']), os.path.basename(path['train']))
    elif data_mode == 'test':
        daio.path_builder(path['log'])
        daio.path_builder(path['test'])
        log_name = 'log_test_{}.txt'.format(os.path.basename(path['test']))
    log_file_path = os.path.join(path['log'], log_name)

    logger = logging.getLogger("AI_COVID19")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger