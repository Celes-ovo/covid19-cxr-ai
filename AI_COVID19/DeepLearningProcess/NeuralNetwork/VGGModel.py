from AI_COVID19.init import *

class VGG:
    def __init__(self, par_dic):
        # parameters
        self.input_size = par_dic['input_size']
        self.block_num = par_dic['block_num']
        self.layer_num = par_dic['layer_num']
        self.drop_out = par_dic['drop_out']
        self.creg_weight = par_dic['conv_reg_weight']
        self.dreg_weight = par_dic['dens_reg_weight']
        self.dens_count = par_dic['dens_count']
        self.output_count = par_dic['output_count']
        self.conv_act = par_dic['conv_act']
        self.dens_act = par_dic['dens_act']
        self.output_act = par_dic['output_act']
        self.conv_str = par_dic['conv_str']
        self.pool_str = par_dic['pool_str']
        self.dens_num = par_dic['dens_num']

    def __call__(self, model_name, array_dim):
        # VGG Free Model
        if model_name == 'VGGFree' and array_dim == '2d':
            inputs = Input(shape=(self.input_size, self.input_size, 1))
            block = layers.Conv2D(32, self.conv_str, activation=self.conv_act, padding='same', kernel_initializer='he_normal')(inputs)
            for i in range(self.block_num):
                block = self.conv_block_2d(block, self.layer_num, [32*(2**i), self.conv_str, self.conv_act, self.pool_str, self.creg_weight])
            dens = self.output_block(block, self.dens_num, self.dens_count, self.dreg_weight, self.flat_act, self.drop_out)
            outputs = layers.Dense(self.output_count, activation=self.output_act)(dens)
            model = Model(inputs, outputs)
            return model

        if model_name == 'VGGFree' and array_dim == '3d':
            inputs = Input(shape=(self.input_size, self.input_size, 1))
            block = layers.Conv3D(32, self.conv_str, activation=self.conv_act, padding='same', kernel_initializer='he_normal')(inputs)
            for i in range(self.block_num):
                block = self.conv_block_3d(block, self.layer_num, [32 * (2 ** i), self.conv_str, self.conv_act, self.pool_str, self.creg_weight])
            dens = self.output_block(block, self.dens_num, self.dens_count, self.dreg_weight, self.flat_act, self.drop_out)
            outputs = layers.Dense(self.output_count, activation=self.output_act)(dens)
            model = Model(inputs, outputs)
            return model

        if model_name == 'VGG16' and array_dim == '2d':
            inputs = Input(shape=(self.input_size, self.input_size, 1))
            block1 = self.conv_block_2d(inputs, 2, [64, self.conv_str, self.conv_act, self.pool_str, self.creg_weight])
            block2 = self.conv_block_2d(block1, 2, [128, self.conv_str, self.conv_act, self.pool_str, self.creg_weight])
            block3 = self.conv_block_2d(block2, 3, [256, self.conv_str, self.conv_act, self.pool_str, self.creg_weight])
            block4 = self.conv_block_2d(block3, 3, [512, self.conv_str, self.conv_act, self.pool_str, self.creg_weight])
            block5 = self.conv_block_2d(block4, 3, [512, self.conv_str, self.conv_act, self.pool_str, self.creg_weight])
            dens = self.output_block(block5, self.dens_num, self.dens_count, self.dreg_weight, self.dens_act, self.drop_out)
            outputs = layers.Dense(self.output_count, activation=self.output_act)(dens)
            model = Model(inputs, outputs)
            return model

        if model_name == 'VGG16' and array_dim == '3d':
            inputs = Input(shape=(self.input_size, self.input_size, self.input_size, 1))
            block1 = self.conv_block_3d(inputs, 2, [64, self.conv_str, self.conv_act, self.pool_str, self.creg_weight])
            block2 = self.conv_block_3d(block1, 2, [128, self.conv_str, self.conv_act, self.pool_str, self.creg_weight])
            block3 = self.conv_block_3d(block2, 3, [256, self.conv_str, self.conv_act, self.pool_str, self.creg_weight])
            block4 = self.conv_block_3d(block3, 3, [512, self.conv_str, self.conv_act, self.pool_str, self.creg_weight])
            block5 = self.conv_block_3d(block4, 3, [512, self.conv_str, self.conv_act, self.pool_str, self.creg_weight])
            dens = self.output_block(block5, self.dens_num, self.dens_count, self.dreg_weight, self.dens_act, self.drop_out)
            outputs = layers.Dense(self.output_count, activation=self.output_act)(dens)
            model = Model(inputs, outputs)
            return model

        if model_name == 'VGG19' and array_dim == '2d':
            inputs = Input(shape=(self.input_size, self.input_size, 1))
            block1 = self.conv_block_2d(inputs, 2, [64, self.conv_str, self.conv_act, self.pool_str, self.creg_weight])
            block2 = self.conv_block_2d(block1, 2, [128, self.conv_str, self.conv_act, self.pool_str, self.creg_weight])
            block3 = self.conv_block_2d(block2, 4, [256, self.conv_str, self.conv_act, self.pool_str, self.creg_weight])
            block4 = self.conv_block_2d(block3, 4, [512, self.conv_str, self.conv_act, self.pool_str, self.creg_weight])
            block5 = self.conv_block_2d(block4, 4, [512, self.conv_str, self.conv_act, self.pool_str, self.creg_weight])
            dens = self.output_block(block5, self.dens_num, self.dens_count, self.dreg_weight, self.dens_act, self.drop_out)
            outputs = layers.Dense(self.output_count, activation=self.output_act)(dens)
            model = Model(inputs, outputs)
            return model

        if model_name == 'VGG19' and array_dim == '3d':
            inputs = Input(shape=(self.input_size, self.input_size, self.input_size, 1))
            block1 = self.conv_block_3d(inputs, 2, [64, self.conv_str, self.conv_act, self.pool_str, self.creg_weight])
            block2 = self.conv_block_3d(block1, 2, [128, self.conv_str, self.conv_act, self.pool_str, self.creg_weight])
            block3 = self.conv_block_3d(block2, 4, [256, self.conv_str, self.conv_act, self.pool_str, self.creg_weight])
            block4 = self.conv_block_3d(block3, 4, [512, self.conv_str, self.conv_act, self.pool_str, self.creg_weight])
            block5 = self.conv_block_3d(block4, 4, [512, self.conv_str, self.conv_act, self.pool_str, self.creg_weight])
            dens = self.output_block(block5, self.dens_num, self.dens_count, self.dreg_weight, self.dens_act, self.drop_out)
            outputs = layers.Dense(self.output_count, activation=self.output_act)(dens)
            model = Model(inputs, outputs)
            return model

    def conv_block_2d(self, lr_conv, lr_num, par_list):
        # parameter
        conv_size = par_list[0]
        conv_str = par_list[1]
        conv_act = par_list[2]
        pool_str = par_list[3]
        reg_weight = par_list[4]
        # code
        for i in range(lr_num):
            lr_conv = layers.Conv2D(conv_size, conv_str, activation=None, padding='same',
                                    kernel_regularizer=reg_weight, kernel_initializer='he_normal')(lr_conv)
            lr_conv = layers.BatchNormalization(axis=-1)(lr_conv)
            lr_conv = layers.Activation(conv_act)(lr_conv)
        lr_pool = layers.MaxPooling2D(pool_size=pool_str)(lr_conv)
        return lr_pool

    def conv_block_3d(self, lr_input, lr_num, par_list):
        # parameter
        conv_size = par_list[0]
        conv_str = par_list[1]
        conv_act = par_list[2]
        pool_str = par_list[3]
        reg_weight = par_list[4]
        # code
        for i in range(lr_num):
            lr_conv = layers.Conv3D(conv_size, conv_str, activation=None, padding='same',
                                    kernel_regularizer=reg_weight, kernel_initializer='he_normal')(lr_input)
            lr_conv = layers.BatchNormalization(axis=-1)(lr_conv)
            lr_conv = layers.Activation(conv_act)(lr_conv)
        lr_pool = layers.MaxPooling3D(pool_size=(pool_str, pool_str, pool_str))(lr_conv)
        return lr_pool

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