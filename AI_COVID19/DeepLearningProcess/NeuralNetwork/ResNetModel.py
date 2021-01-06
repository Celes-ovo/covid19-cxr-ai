from AI_COVID19.init import *

class ResNet:
    def __init__(self, par_dic):
        # parameters
        self.input_size = par_dic['input_size']
        self.channel_num = par_dic['channel_num']
        self.creg_weight = par_dic['conv_reg_weight']
        self.dreg_weight = par_dic['dens_reg_weight']
        self.act_func = par_dic['act_func']
        self.dens_num = par_dic['dens_num']
        self.dens_count = par_dic['dens_count']
        self.drop_rate = par_dic['drop_rate']
        self.output_count = par_dic['output_count']
        self.output_act = par_dic['output_act']

    def __call__(self, model_name, array_dim):
        if model_name == 'ResNet50' and array_dim == '2d':
            inputs = layers.Input(shape=(self.input_size, self.input_size, self.channel_num))
            conv1 = self.conv1_block(inputs, self.creg_weight)  # Blcok1
            conv2 = self.res_conv_block(conv1, 64, 3, self.creg_weight, self.act_func, mode='hold')  # Block2
            conv3 = self.res_conv_block(conv2, 128, 4, self.creg_weight, self.act_func)  # Block3
            conv4 = self.res_conv_block(conv3, 256, 6, self.creg_weight, self.act_func)  # Block4
            conv5 = self.res_conv_block(conv4, 512, 3, self.creg_weight, self.act_func)  # Block5
            dens = self.output_block(conv5, self.dens_num, self.dens_count, self.dreg_weight, self.act_func, self.drop_rate)
            outputs = layers.Dense(self.output_count, activation=self.output_act)(dens)
            model = Model(inputs, outputs)
            return model

        if model_name == 'ResNet101' and array_dim == '2d':
            inputs = layers.Input(shape=(self.input_size, self.input_size, self.channel_num))
            conv1 = self.conv1_block(inputs, self.creg_weight)  # Blcok1
            conv2 = self.res_conv_block(conv1, 64, 3, self.creg_weight, self.act_func, mode='hold')  # Block2
            conv3 = self.res_conv_block(conv2, 128, 4, self.creg_weight, self.act_func)  # Block3
            conv4 = self.res_conv_block(conv3, 256, 23, self.creg_weight, self.act_func)  # Block4
            conv5 = self.res_conv_block(conv4, 512, 3, self.creg_weight, self.act_func)  # Block5
            dens = self.output_block(conv5, self.dens_num, self.dens_count, self.dreg_weight, self.act_func, self.drop_rate)
            outputs = layers.Dense(self.output_count, activation=self.output_act)(dens)
            model = Model(inputs, outputs)
            return model

        if model_name == 'ResNet152' and array_dim == '2d':
            inputs = layers.Input(shape=(self.input_size, self.input_size, self.channel_num))
            conv1 = self.conv1_block(inputs, self.creg_weight)  # Blcok1
            conv2 = self.res_conv_block(conv1, 64, 3, self.creg_weight, self.act_func, mode='hold')  # Block2
            conv3 = self.res_conv_block(conv2, 128, 8, self.creg_weight, self.act_func)  # Block3
            conv4 = self.res_conv_block(conv3, 256, 36, self.creg_weight, self.act_func)  # Block4
            conv5 = self.res_conv_block(conv4, 512, 3, self.creg_weight, self.act_func)  # Block5
            dens = self.output_block(conv5, self.dens_num, self.dens_count, self.dreg_weight, self.act_func, self.drop_rate)
            outputs = layers.Dense(self.output_count, activation=self.output_act)(dens)
            model = Model(inputs, outputs)
            return model

    def conv1_block(self, lr_conv, reg_weight):
        # layer blcok
        lr_conv = layers.ZeroPadding2D(3)(lr_conv)
        lr_conv = layers.Conv2D(64, 7, 2, activation=None, padding='valid', kernel_initializer='he_normal',
                                kernel_regularizer=reg_weight)(lr_conv)
        lr_conv = layers.BatchNormalization(axis=-1)(lr_conv)
        lr_conv = layers.Activation('relu')(lr_conv)
        lr_conv = layers.ZeroPadding2D(1)(lr_conv)
        lr_conv = layers.MaxPool2D(3, 2, padding='valid')(lr_conv)
        return lr_conv

    def res_conv_block(self, lr_io, ker_size, block_num, reg_weight, act_func, mode=None):
        for i in range(block_num):
            if mode == 'hold':
                fstr = 1
            else:
                fstr = 2
            # layer block
            if i == 0:
                lr_conv1 = layers.Conv2D(ker_size, 1, fstr, padding='same', kernel_initializer='he_normal',
                                         kernel_regularizer=reg_weight)(lr_io)
            else:
                lr_conv1 = layers.Conv2D(ker_size, 1, 1, padding='same', kernel_initializer='he_normal',
                                         kernel_regularizer=reg_weight)(lr_io)
            lr_conv1 = layers.BatchNormalization(axis=-1)(lr_conv1)
            lr_conv1 = layers.Activation(act_func)(lr_conv1)
            lr_conv2 = layers.Conv2D(ker_size, 3, 1, padding='same', kernel_initializer='he_normal',
                                     kernel_regularizer=reg_weight)(lr_conv1)
            lr_conv2 = layers.BatchNormalization(axis=-1)(lr_conv2)
            lr_conv2 = layers.Activation(act_func)(lr_conv2)
            lr_conv3 = layers.Conv2D(4 * ker_size, 1, 1, padding='same', kernel_initializer='he_normal',
                                     kernel_regularizer=reg_weight)(lr_conv2)
            lr_conv3 = layers.BatchNormalization(axis=-1)(lr_conv3)
            if i == 0:
                lr_conv0 = layers.Conv2D(4 * ker_size, 1, fstr, padding='same', kernel_initializer='he_normal',
                                         kernel_regularizer=reg_weight)(lr_io)
                lr_conv0 = layers.BatchNormalization(axis=-1)(lr_conv0)
                lr_add = layers.Add()([lr_conv0, lr_conv3])
            else:
                lr_add = layers.Add()([lr_io, lr_conv3])
            lr_io = layers.Activation(act_func)(lr_add)
        return lr_io

    def output_block(self, lr_dense, block_num, flat_count, reg_weight, act_func, drop_rate):
        lr_dense = layers.Flatten()(lr_dense)
        lr_dense = layers.Dropout(drop_rate)(lr_dense)
        for i in range(block_num):
            lr_dense = layers.Dense(flat_count[i], kernel_regularizer=reg_weight,
                                    activation=None)(lr_dense)
            lr_dense = layers.BatchNormalization(axis=-1)(lr_dense)
            lr_dense = layers.Activation(act_func)(lr_dense)
            lr_dense = layers.Dropout(drop_rate)(lr_dense)
        return lr_dense
