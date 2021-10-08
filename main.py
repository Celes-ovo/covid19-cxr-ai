##################
# Library Import #
##################
from AI_COVID19.init import *  # All Library Import
import AI_COVID19.config as conf  # All Parameter Configuration
import AI_COVID19.log as ailog   # Logging & Path Making Code
from AI_COVID19.ImageProcess.npz_converter import NpzConverter as proc_npzc   # Data IO Processing
from AI_COVID19.ImageProcess.pre_processing import PreProcessing as proc_prep   # Image Pre-Processing
from AI_COVID19.DeepLearningProcess.train_cls import TrainCLS as proc_train   # Train Processing

################
# Path setting #
################
path = conf.path_param_set['local']  # 'server' or 'local' choose

###################
# Process Running #
###################
if __name__ == '__main__':
    ### Log & Output Setup ###
    logger_ai = ailog.logger_setting(path['output_path'], 'train')   # Log Setting

    ### All Process ###
    # proc_npzc(conf.prep_param_set['npz_convert'])(path['input_path']['dataset'], path['output_path']['prep'], 'org', logger_ai)
    proc_prep(conf.prep_param_set['pre_process'])(path['input_path']['prep'], path['output_path']['prep'], 'prep', logger_ai)
    # proc_train(conf.train_param_set['cls_vgg'], conf.network_param_set['vgg'])(path['input_path']['train'],
    #                                                                            path['output_path']['train'],
    #                                                                            'VGG19_1', logger_ai)
    