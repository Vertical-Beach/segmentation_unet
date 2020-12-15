import tensorflow as tf
from util import loader as ld
import numpy as np
class FPN:
    def __init__(self, size=(128, 128), l2_reg=None):
        self.model = self.create_model(size, l2_reg)
    @staticmethod
    def make_var(name, shape):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, trainable=True)
    @staticmethod
    def conv(
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding='SAME',
             group=1,
             biased=True):
        conved = tf.layers.conv2d(
            inputs=input,
            filters=c_o,
            kernel_size=[k_h, k_w],
            strides=[s_h, s_w],
            padding="same",
            activation=None,
            use_bias = False, #Test
            kernel_regularizer=None
        )
        return conved
        # Get the number of channels in the input
        # c_i =int(input.get_shape()[-1])
        # # Verify that the grouping parameter is valid
        # assert c_i % group == 0
        # assert c_o % group == 0
        # # Convolution for a given input and kernel
        # with tf.variable_scope(name) as scope:
        #     conved = tf.nn.conv2d(input, FPN.make_var('weights', [k_h, k_w, c_i, c_o]), [1, s_h, s_w, 1], padding=padding)
        #     if relu:
        #         conved = tf.nn.relu(conved, name=scope.name)
        #     return conved
    @staticmethod
    def deconv(
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding='SAME',
             group=1,
             biased=True):
        conved = tf.layers.conv2d_transpose(
            inputs=input,
            filters=c_o,
            strides=[s_h, s_w],
            kernel_size=[k_h, k_w],
            padding='same',
            activation=None,
            use_bias = False, #Test
            kernel_regularizer=None
        )
        return conved
        ######## workaround
        # # if padding is None:
        # #     padding = 'SAME'     
        # # Verify that the padding is acceptable
        # # Get the number of channels in the input
        # c_i = input.get_shape()[-1]
        # # Verify that the grouping parameter is valid
        # assert c_i % group == 0
        # assert c_o % group == 0
        # # Convolution for a given input and kernel
        # stride_shape = [1, s_h, s_w, 1]
        # kernel_shape = [k_h, k_w, c_i, c_o]
        # input_shape = input.get_shape().as_list()
        # out_shape = []
        # for i in range(len(input.get_shape())-1):
        #     if i == 0:
        #         if input_shape[i] is None:
        #             out_shape.append(None)
        #         else:
        #             out_shape.append(int(input_shape[i]))
        #     else:
        #         kernel_i= int(kernel_shape[i])
        #         stride_i = int(stride_shape[i])
        #         input_i = int(input_shape[i])
        #         o = stride_i * (input_i - 1) + int(kernel_shape[0]) - 2 * padding
        #         out_shape.append(o)
        # out_shape.append(c_o)
        #         # if padding == 'SAME':   
        #         #     out_shape.append(int(np.ceil(float(input_i) / float(stride_i) )))
        #         # else:
        #         #     out_shape.append(int(np.ceil(float(input_i - kernel_i + 1) / float(stride_i) )))
        # print(input_shape, name, out_shape)
        #     #out_shape.append(Math.floor((input_i + 2 * pad_i - kernel_i) / float(stride_i) + 1))
        # with tf.variable_scope(name) as scope:
        #     if padding == 1:
        #         padding_str = 'SAME'
        #     else:
        #         padding_str = 'VALID'
        #     output = tf.nn.conv2d_transpose(input, FPN.make_var('weights', shape=[k_h, k_w, c_i, c_o]), output_shape=out_shape, strides=stride_shape, padding=padding_str)
        #     if relu:
        #         # ReLU non-linearity
        #         output = tf.nn.relu(output, name=scope.name)
        #     return output
    @staticmethod
    def max_pool(input, k_h, k_w, s_h, s_w, name, padding='SAME'):
        return tf.layers.max_pooling2d(inputs=input, pool_size=[k_h, k_w], strides=[s_h, s_w], padding='SAME')
        # return tf.nn.max_pool(input,
        #                       ksize=[1, k_h, k_w, 1],
        #                       strides=[1, s_h, s_w, 1],
        #                       padding=padding,
        #                       name=name)
    @staticmethod
    def concat(inputs, axis, name):
        return tf.concat(axis=axis, values=inputs, name=name)
    @staticmethod
    def add(inputs, name):
        #for DPU
        # return tf.add_n(inputs, name=name)
        assert(len(inputs) == 2)
        return tf.add(inputs[0], inputs[1])
    @staticmethod
    def batch_normalization(input, name, scale_offset=True, relu=False):
        is_training = False #workaround
        normalized = tf.layers.batch_normalization(
            inputs=input,
            axis=-1,
            momentum=0.9,
            epsilon=0.001,
            center=True,
            scale=True,
            training=is_training,
        )
        if relu:
            normalized = tf.nn.relu(normalized)
        return normalized
    @staticmethod
    def create_model(size, l2_reg):
        num_classes = len(ld.DataSet.CATEGORY)
        inputs = tf.placeholder(tf.float32, [None, size[0], size[1], 3])
        teacher = tf.placeholder(tf.float32, [None, size[0], size[1], num_classes])
        is_training = tf.placeholder(tf.bool)
        
        conv1_7x7_s2 = FPN.conv(inputs, 7, 7, 64, 2, 2, relu=False, name='conv1_7x7_s2')
        conv1_7x7_s2_BatchNorm = FPN.batch_normalization(conv1_7x7_s2, scale_offset=False, relu=True, name='conv1_7x7_s2_BatchNorm')
        pool1_3x3_s2 = FPN.max_pool(conv1_7x7_s2_BatchNorm, 3, 3, 2, 2, name='pool1_3x3_s2')
        conv2_3x3_reduce = FPN.conv(pool1_3x3_s2, 1, 1, 64, 1, 1, relu=False, name='conv2_3x3_reduce')
        conv2_3x3_reduce_BatchNorm = FPN.batch_normalization(conv2_3x3_reduce, scale_offset=False, relu=True, name='conv2_3x3_reduce_BatchNorm')
        conv2_3x3 = FPN.conv(conv2_3x3_reduce_BatchNorm, 3, 3, 192, 1, 1, relu=False, name='conv2_3x3')
        conv2_3x3_BatchNorm = FPN.batch_normalization(conv2_3x3, scale_offset=False, relu=True, name='conv2_3x3_BatchNorm')
        pool2_3x3_s2 = FPN.max_pool(conv2_3x3_BatchNorm, 3, 3, 2, 2, name='pool2_3x3_s2')
        inception_3a_1x1 = FPN.conv(pool2_3x3_s2, 1, 1, 64, 1, 1, relu=False, name='inception_3a_1x1')
        inception_3a_1x1_BatchNorm = FPN.batch_normalization(inception_3a_1x1, scale_offset=False, relu=True, name='inception_3a_1x1_BatchNorm')

        inception_3a_3x3_reduce = FPN.conv(pool2_3x3_s2, 1, 1, 96, 1, 1, relu=False, name='inception_3a_3x3_reduce')
        inception_3a_3x3_reduce_BatchNorm = FPN.batch_normalization(inception_3a_3x3_reduce, scale_offset=False, relu=True, name='inception_3a_3x3_reduce_BatchNorm')
        inception_3a_3x3 = FPN.conv(inception_3a_3x3_reduce_BatchNorm, 3, 3, 128, 1, 1, relu=False, name='inception_3a_3x3')
        inception_3a_3x3_BatchNorm = FPN.batch_normalization(inception_3a_3x3, scale_offset=False, relu=True, name='inception_3a_3x3_BatchNorm')

        inception_3a_5x5_reduce = FPN.conv(pool2_3x3_s2, 1, 1, 16, 1, 1, relu=False, name='inception_3a_5x5_reduce')
        inception_3a_5x5_reduce_BatchNorm = FPN.batch_normalization(inception_3a_5x5_reduce, scale_offset=False, relu=True, name='inception_3a_5x5_reduce_BatchNorm')
        inception_3a_5x5 = FPN.conv(inception_3a_5x5_reduce_BatchNorm, 5, 5, 32, 1, 1, relu=False, name='inception_3a_5x5')
        inception_3a_5x5_BatchNorm = FPN.batch_normalization(inception_3a_5x5, scale_offset=False, relu=True, name='inception_3a_5x5_BatchNorm')

        inception_3a_pool = FPN.max_pool(pool2_3x3_s2, 3, 3, 1, 1, name='inception_3a_pool')
        inception_3a_pool_proj = FPN.conv(inception_3a_pool, 1, 1, 32, 1, 1, relu=False, name='inception_3a_pool_proj')
        inception_3a_pool_proj_BatchNorm = FPN.batch_normalization(inception_3a_pool_proj, scale_offset=False, relu=True, name='inception_3a_pool_proj_BatchNorm')

        inception_3a_output = FPN.concat([inception_3a_1x1_BatchNorm,inception_3a_3x3_BatchNorm,inception_3a_5x5_BatchNorm,inception_3a_pool_proj_BatchNorm], 3, name='inception_3a_output')
        inception_3b_1x1 = FPN.conv(inception_3a_output, 1, 1, 128, 1, 1, relu=False, name='inception_3b_1x1')
        inception_3b_1x1_BatchNorm = FPN.batch_normalization(inception_3b_1x1, scale_offset=False, relu=True, name='inception_3b_1x1_BatchNorm')

        inception_3b_3x3_reduce = FPN.conv(inception_3a_output, 1, 1, 128, 1, 1, relu=False, name='inception_3b_3x3_reduce')
        inception_3b_3x3_reduce_BatchNorm = FPN.batch_normalization(inception_3b_3x3_reduce, scale_offset=False, relu=True, name='inception_3b_3x3_reduce_BatchNorm')
        inception_3b_3x3 = FPN.conv(inception_3b_3x3_reduce_BatchNorm, 3, 3, 192, 1, 1, relu=False, name='inception_3b_3x3')
        inception_3b_3x3_BatchNorm = FPN.batch_normalization(inception_3b_3x3, scale_offset=False, relu=True, name='inception_3b_3x3_BatchNorm')

        inception_3b_5x5_reduce = FPN.conv(inception_3a_output, 1, 1, 32, 1, 1, relu=False, name='inception_3b_5x5_reduce')
        inception_3b_5x5_reduce_BatchNorm = FPN.batch_normalization(inception_3b_5x5_reduce, scale_offset=False, relu=True, name='inception_3b_5x5_reduce_BatchNorm')
        inception_3b_5x5 = FPN.conv(inception_3b_5x5_reduce_BatchNorm, 5, 5, 96, 1, 1, relu=False, name='inception_3b_5x5')
        inception_3b_5x5_BatchNorm = FPN.batch_normalization(inception_3b_5x5, scale_offset=False, relu=True, name='inception_3b_5x5_BatchNorm')

        inception_3b_pool = FPN.max_pool(inception_3a_output, 3, 3, 1, 1, name='inception_3b_pool')
        inception_3b_pool_proj = FPN.conv(inception_3b_pool, 1, 1, 64, 1, 1, relu=False, name='inception_3b_pool_proj')
        inception_3b_pool_proj_BatchNorm = FPN.batch_normalization(inception_3b_pool_proj, scale_offset=False, relu=True, name='inception_3b_pool_proj_BatchNorm')

        inception_3b_output = FPN.concat([inception_3b_1x1_BatchNorm,inception_3b_3x3_BatchNorm,inception_3b_5x5_BatchNorm,inception_3b_pool_proj_BatchNorm], 3, name='inception_3b_output')
        pool3_3x3_s2 = FPN.max_pool(inception_3b_output, 3, 3, 2, 2, name='pool3_3x3_s2')
        inception_4a_1x1 = FPN.conv(pool3_3x3_s2, 1, 1, 192, 1, 1, relu=False, name='inception_4a_1x1')
        inception_4a_1x1_BatchNorm = FPN.batch_normalization(inception_4a_1x1, scale_offset=False, relu=True, name='inception_4a_1x1_BatchNorm')

        inception_4a_3x3_reduce = FPN.conv(pool3_3x3_s2, 1, 1, 96, 1, 1, relu=False, name='inception_4a_3x3_reduce')
        inception_4a_3x3_reduce_BatchNorm = FPN.batch_normalization(inception_4a_3x3_reduce, scale_offset=False, relu=True, name='inception_4a_3x3_reduce_BatchNorm')
        inception_4a_3x3 = FPN.conv(inception_4a_3x3_reduce_BatchNorm, 3, 3, 208, 1, 1, relu=False, name='inception_4a_3x3')
        inception_4a_3x3_BatchNorm = FPN.batch_normalization(inception_4a_3x3, scale_offset=False, relu=True, name='inception_4a_3x3_BatchNorm')

        inception_4a_5x5_reduce = FPN.conv(pool3_3x3_s2, 1, 1, 16, 1, 1, relu=False, name='inception_4a_5x5_reduce')
        inception_4a_5x5_reduce_BatchNorm = FPN.batch_normalization(inception_4a_5x5_reduce, scale_offset=False, relu=True, name='inception_4a_5x5_reduce_BatchNorm')
        inception_4a_5x5 = FPN.conv(inception_4a_5x5_reduce_BatchNorm, 5, 5, 48, 1, 1, relu=False, name='inception_4a_5x5')
        inception_4a_5x5_BatchNorm = FPN.batch_normalization(inception_4a_5x5, scale_offset=False, relu=True, name='inception_4a_5x5_BatchNorm')

        inception_4a_pool = FPN.max_pool(pool3_3x3_s2, 3, 3, 1, 1, name='inception_4a_pool')
        inception_4a_pool_proj = FPN.conv(inception_4a_pool, 1, 1, 64, 1, 1, relu=False, name='inception_4a_pool_proj')
        inception_4a_pool_proj_BatchNorm = FPN.batch_normalization(inception_4a_pool_proj, scale_offset=False, relu=True, name='inception_4a_pool_proj_BatchNorm')

        inception_4a_output = FPN.concat([inception_4a_1x1_BatchNorm,inception_4a_3x3_BatchNorm,inception_4a_5x5_BatchNorm,inception_4a_pool_proj_BatchNorm], 3, name='inception_4a_output')
        inception_4b_1x1 = FPN.conv(inception_4a_output, 1, 1, 160, 1, 1, relu=False, name='inception_4b_1x1')
        inception_4b_1x1_BatchNorm = FPN.batch_normalization(inception_4b_1x1, scale_offset=False, relu=True, name='inception_4b_1x1_BatchNorm')

        inception_4b_3x3_reduce = FPN.conv(inception_4a_output, 1, 1, 112, 1, 1, relu=False, name='inception_4b_3x3_reduce')
        inception_4b_3x3_reduce_BatchNorm = FPN.batch_normalization(inception_4b_3x3_reduce, scale_offset=False, relu=True, name='inception_4b_3x3_reduce_BatchNorm')
        inception_4b_3x3 = FPN.conv(inception_4b_3x3_reduce_BatchNorm, 3, 3, 224, 1, 1, relu=False, name='inception_4b_3x3')
        inception_4b_3x3_BatchNorm = FPN.batch_normalization(inception_4b_3x3, scale_offset=False, relu=True, name='inception_4b_3x3_BatchNorm')

        inception_4b_5x5_reduce = FPN.conv(inception_4a_output, 1, 1, 24, 1, 1, relu=False, name='inception_4b_5x5_reduce')
        inception_4b_5x5_reduce_BatchNorm = FPN.batch_normalization(inception_4b_5x5_reduce, scale_offset=False, relu=True, name='inception_4b_5x5_reduce_BatchNorm')
        inception_4b_5x5 = FPN.conv(inception_4b_5x5_reduce_BatchNorm, 5, 5, 64, 1, 1, relu=False, name='inception_4b_5x5')
        inception_4b_5x5_BatchNorm = FPN.batch_normalization(inception_4b_5x5, scale_offset=False, relu=True, name='inception_4b_5x5_BatchNorm')

        inception_4b_pool = FPN.max_pool(inception_4a_output, 3, 3, 1, 1, name='inception_4b_pool')
        inception_4b_pool_proj = FPN.conv(inception_4b_pool, 1, 1, 64, 1, 1, relu=False, name='inception_4b_pool_proj')
        inception_4b_pool_proj_BatchNorm = FPN.batch_normalization(inception_4b_pool_proj, scale_offset=False, relu=True, name='inception_4b_pool_proj_BatchNorm')

        inception_4b_output = FPN.concat([inception_4b_1x1_BatchNorm,inception_4b_3x3_BatchNorm,inception_4b_5x5_BatchNorm,inception_4b_pool_proj_BatchNorm], 3, name='inception_4b_output')
        inception_4c_1x1 = FPN.conv(inception_4b_output, 1, 1, 128, 1, 1, relu=False, name='inception_4c_1x1')
        inception_4c_1x1_BatchNorm = FPN.batch_normalization(inception_4c_1x1, scale_offset=False, relu=True, name='inception_4c_1x1_BatchNorm')

        inception_4c_3x3_reduce = FPN.conv(inception_4b_output, 1, 1, 128, 1, 1, relu=False, name='inception_4c_3x3_reduce')
        inception_4c_3x3_reduce_BatchNorm = FPN.batch_normalization(inception_4c_3x3_reduce, scale_offset=False, relu=True, name='inception_4c_3x3_reduce_BatchNorm')
        inception_4c_3x3 = FPN.conv(inception_4c_3x3_reduce_BatchNorm, 3, 3, 256, 1, 1, relu=False, name='inception_4c_3x3')
        inception_4c_3x3_BatchNorm = FPN.batch_normalization(inception_4c_3x3, scale_offset=False, relu=True, name='inception_4c_3x3_BatchNorm')

        inception_4c_5x5_reduce = FPN.conv(inception_4b_output, 1, 1, 24, 1, 1, relu=False, name='inception_4c_5x5_reduce')
        inception_4c_5x5_reduce_BatchNorm = FPN.batch_normalization(inception_4c_5x5_reduce, scale_offset=False, relu=True, name='inception_4c_5x5_reduce_BatchNorm')
        inception_4c_5x5 = FPN.conv(inception_4c_5x5_reduce_BatchNorm, 5, 5, 64, 1, 1, relu=False, name='inception_4c_5x5')
        inception_4c_5x5_BatchNorm = FPN.batch_normalization(inception_4c_5x5, scale_offset=False, relu=True, name='inception_4c_5x5_BatchNorm')

        inception_4c_pool = FPN.max_pool(inception_4b_output, 3, 3, 1, 1, name='inception_4c_pool')
        inception_4c_pool_proj = FPN.conv(inception_4c_pool, 1, 1, 64, 1, 1, relu=False, name='inception_4c_pool_proj')
        inception_4c_pool_proj_BatchNorm = FPN.batch_normalization(inception_4c_pool_proj, scale_offset=False, relu=True, name='inception_4c_pool_proj_BatchNorm')

        inception_4c_output = FPN.concat([inception_4c_1x1_BatchNorm,inception_4c_3x3_BatchNorm,inception_4c_5x5_BatchNorm,inception_4c_pool_proj_BatchNorm], 3, name='inception_4c_output')
        inception_4d_1x1 = FPN.conv(inception_4c_output, 1, 1, 112, 1, 1, relu=False, name='inception_4d_1x1')
        inception_4d_1x1_BatchNorm = FPN.batch_normalization(inception_4d_1x1, scale_offset=False, relu=True, name='inception_4d_1x1_BatchNorm')

        inception_4d_3x3_reduce = FPN.conv(inception_4c_output, 1, 1, 144, 1, 1, relu=False, name='inception_4d_3x3_reduce')
        inception_4d_3x3_reduce_BatchNorm = FPN.batch_normalization(inception_4d_3x3_reduce, scale_offset=False, relu=True, name='inception_4d_3x3_reduce_BatchNorm')
        inception_4d_3x3 = FPN.conv(inception_4d_3x3_reduce_BatchNorm, 3, 3, 288, 1, 1, relu=False, name='inception_4d_3x3')
        inception_4d_3x3_BatchNorm = FPN.batch_normalization(inception_4d_3x3, scale_offset=False, relu=True, name='inception_4d_3x3_BatchNorm')

        inception_4d_5x5_reduce = FPN.conv(inception_4c_output, 1, 1, 32, 1, 1, relu=False, name='inception_4d_5x5_reduce')
        inception_4d_5x5_reduce_BatchNorm = FPN.batch_normalization(inception_4d_5x5_reduce, scale_offset=False, relu=True, name='inception_4d_5x5_reduce_BatchNorm')
        inception_4d_5x5 = FPN.conv(inception_4d_5x5_reduce_BatchNorm, 5, 5, 64, 1, 1, relu=False, name='inception_4d_5x5')
        inception_4d_5x5_BatchNorm = FPN.batch_normalization(inception_4d_5x5, scale_offset=False, relu=True, name='inception_4d_5x5_BatchNorm')

        inception_4d_pool = FPN.max_pool(inception_4c_output, 3, 3, 1, 1, name='inception_4d_pool')
        inception_4d_pool_proj = FPN.conv(inception_4d_pool, 1, 1, 64, 1, 1, relu=False, name='inception_4d_pool_proj')
        inception_4d_pool_proj_BatchNorm = FPN.batch_normalization(inception_4d_pool_proj, scale_offset=False, relu=True, name='inception_4d_pool_proj_BatchNorm')

        inception_4d_output = FPN.concat([inception_4d_1x1_BatchNorm,inception_4d_3x3_BatchNorm,inception_4d_5x5_BatchNorm,inception_4d_pool_proj_BatchNorm], 3, name='inception_4d_output')
        inception_4e_1x1 = FPN.conv(inception_4d_output, 1, 1, 256, 1, 1, relu=False, name='inception_4e_1x1')
        inception_4e_1x1_BatchNorm = FPN.batch_normalization(inception_4e_1x1, scale_offset=False, relu=True, name='inception_4e_1x1_BatchNorm')

        inception_4e_3x3_reduce = FPN.conv(inception_4d_output, 1, 1, 160, 1, 1, relu=False, name='inception_4e_3x3_reduce')
        inception_4e_3x3_reduce_BatchNorm = FPN.batch_normalization(inception_4e_3x3_reduce, scale_offset=False, relu=True, name='inception_4e_3x3_reduce_BatchNorm')
        inception_4e_3x3 = FPN.conv(inception_4e_3x3_reduce_BatchNorm, 3, 3, 320, 1, 1, relu=False, name='inception_4e_3x3')
        inception_4e_3x3_BatchNorm = FPN.batch_normalization(inception_4e_3x3, scale_offset=False, relu=True, name='inception_4e_3x3_BatchNorm')

        inception_4e_5x5_reduce = FPN.conv(inception_4d_output, 1, 1, 32, 1, 1, relu=False, name='inception_4e_5x5_reduce')
        inception_4e_5x5_reduce_BatchNorm = FPN.batch_normalization(inception_4e_5x5_reduce, scale_offset=False, relu=True, name='inception_4e_5x5_reduce_BatchNorm')
        inception_4e_5x5 = FPN.conv(inception_4e_5x5_reduce_BatchNorm, 5, 5, 128, 1, 1, relu=False, name='inception_4e_5x5')
        inception_4e_5x5_BatchNorm = FPN.batch_normalization(inception_4e_5x5, scale_offset=False, relu=True, name='inception_4e_5x5_BatchNorm')

        inception_4e_pool = FPN.max_pool(inception_4d_output, 3, 3, 1, 1, name='inception_4e_pool')
        inception_4e_pool_proj = FPN.conv(inception_4e_pool, 1, 1, 128, 1, 1, relu=False, name='inception_4e_pool_proj')
        inception_4e_pool_proj_BatchNorm = FPN.batch_normalization(inception_4e_pool_proj, scale_offset=False, relu=True, name='inception_4e_pool_proj_BatchNorm')

        inception_4e_output = FPN.concat([inception_4e_1x1_BatchNorm,inception_4e_3x3_BatchNorm,inception_4e_5x5_BatchNorm,inception_4e_pool_proj_BatchNorm], 3, name='inception_4e_output')
        pool4_3x3_s2 = FPN.max_pool(inception_4e_output, 3, 3, 2, 2, name='pool4_3x3_s2')
        inception_5a_1x1 = FPN.conv(pool4_3x3_s2, 1, 1, 256, 1, 1, relu=False, name='inception_5a_1x1')
        inception_5a_1x1_BatchNorm = FPN.batch_normalization(inception_5a_1x1, scale_offset=False, relu=True, name='inception_5a_1x1_BatchNorm')

        inception_5a_3x3_reduce = FPN.conv(pool4_3x3_s2, 1, 1, 160, 1, 1, relu=False, name='inception_5a_3x3_reduce')
        inception_5a_3x3_reduce_BatchNorm = FPN.batch_normalization(inception_5a_3x3_reduce, scale_offset=False, relu=True, name='inception_5a_3x3_reduce_BatchNorm')
        inception_5a_3x3 = FPN.conv(inception_5a_3x3_reduce_BatchNorm, 3, 3, 320, 1, 1, relu=False, name='inception_5a_3x3')
        inception_5a_3x3_BatchNorm = FPN.batch_normalization(inception_5a_3x3, scale_offset=False, relu=True, name='inception_5a_3x3_BatchNorm')

        inception_5a_5x5_reduce = FPN.conv(pool4_3x3_s2, 1, 1, 32, 1, 1, relu=False, name='inception_5a_5x5_reduce')
        inception_5a_5x5_reduce_BatchNorm = FPN.batch_normalization(inception_5a_5x5_reduce, scale_offset=False, relu=True, name='inception_5a_5x5_reduce_BatchNorm')
        inception_5a_5x5 = FPN.conv(inception_5a_5x5_reduce_BatchNorm, 5, 5, 128, 1, 1, relu=False, name='inception_5a_5x5')
        inception_5a_5x5_BatchNorm = FPN.batch_normalization(inception_5a_5x5, scale_offset=False, relu=True, name='inception_5a_5x5_BatchNorm')

        inception_5a_pool = FPN.max_pool(pool4_3x3_s2, 3, 3, 1, 1, name='inception_5a_pool')
        inception_5a_pool_proj = FPN.conv(inception_5a_pool, 1, 1, 128, 1, 1, relu=False, name='inception_5a_pool_proj')
        inception_5a_pool_proj_BatchNorm = FPN.batch_normalization(inception_5a_pool_proj, scale_offset=False, relu=True, name='inception_5a_pool_proj_BatchNorm')

        inception_5a_output = FPN.concat([inception_5a_1x1_BatchNorm,inception_5a_3x3_BatchNorm,inception_5a_5x5_BatchNorm,inception_5a_pool_proj_BatchNorm], 3, name='inception_5a_output')
        inception_5b_1x1 = FPN.conv(inception_5a_output, 1, 1, 384, 1, 1, relu=False, name='inception_5b_1x1')
        inception_5b_1x1_BatchNorm = FPN.batch_normalization(inception_5b_1x1, scale_offset=False, relu=True, name='inception_5b_1x1_BatchNorm')

        inception_5b_3x3_reduce = FPN.conv(inception_5a_output, 1, 1, 192, 1, 1, relu=False, name='inception_5b_3x3_reduce')
        inception_5b_3x3_reduce_BatchNorm = FPN.batch_normalization(inception_5b_3x3_reduce, scale_offset=False, relu=True, name='inception_5b_3x3_reduce_BatchNorm')
        inception_5b_3x3 = FPN.conv(inception_5b_3x3_reduce_BatchNorm, 3, 3, 384, 1, 1, relu=False, name='inception_5b_3x3')
        inception_5b_3x3_BatchNorm = FPN.batch_normalization(inception_5b_3x3, scale_offset=False, relu=True, name='inception_5b_3x3_BatchNorm')

        inception_5b_5x5_reduce = FPN.conv(inception_5a_output, 1, 1, 48, 1, 1, relu=False, name='inception_5b_5x5_reduce')
        inception_5b_5x5_reduce_BatchNorm = FPN.batch_normalization(inception_5b_5x5_reduce, scale_offset=False, relu=True, name='inception_5b_5x5_reduce_BatchNorm')
        inception_5b_5x5 = FPN.conv(inception_5b_5x5_reduce_BatchNorm, 5, 5, 128, 1, 1, relu=False, name='inception_5b_5x5')
        inception_5b_5x5_BatchNorm = FPN.batch_normalization(inception_5b_5x5, scale_offset=False, relu=True, name='inception_5b_5x5_BatchNorm')

        inception_5b_pool = FPN.max_pool(inception_5a_output, 3, 3, 1, 1, name='inception_5b_pool')
        inception_5b_pool_proj = FPN.conv(inception_5b_pool, 1, 1, 128, 1, 1, relu=False, name='inception_5b_pool_proj')
        inception_5b_pool_proj_BatchNorm = FPN.batch_normalization(inception_5b_pool_proj, scale_offset=False, relu=True, name='inception_5b_pool_proj_BatchNorm')

        inception_5b_output = FPN.concat([inception_5b_1x1_BatchNorm,inception_5b_3x3_BatchNorm,inception_5b_5x5_BatchNorm,inception_5b_pool_proj_BatchNorm], 3, name='inception_5b_output')
        p5 = FPN.conv(inception_5b_output, 1, 1, 32, 1, 1, relu=False, name='p5')
        upsample_p5 = FPN.deconv(p5, 4, 4, 32, 2, 2, padding=1, biased=False, relu=False, name='upsample_p5')

        latlayer_4f = FPN.conv(inception_4e_output, 1, 1, 32, 1, 1, relu=False, name='latlayer_4f')

        add_p4 = FPN.add([upsample_p5,latlayer_4f], name='add_p4')
        toplayer_p4 = FPN.conv(add_p4, 3, 3, 32, 1, 1, relu=False, name='toplayer_p4')
        upsample_p4 = FPN.deconv(toplayer_p4, 4, 4, 32, 2, 2, padding=1, biased=False, relu=False, name='upsample_p4')
        latlayer_3d = FPN.conv(inception_3b_output, 1, 1, 32, 1, 1, relu=False, name='latlayer_3d')

        add_p3 = FPN.add([upsample_p4,latlayer_3d], name='add_p3')
        toplayer_p3 = FPN.conv(add_p3, 3, 3, 32, 1, 1, relu=False, name='toplayer_p3')
        upsample_p3 = FPN.deconv(toplayer_p3, 4, 4, 32, 2, 2, padding=1, biased=False, relu=False, name='upsample_p3')

        latlayer_2c = FPN.conv(conv2_3x3_BatchNorm, 1, 1, 32, 1, 1, relu=False, name='latlayer_2c')

        add_p2 = FPN.add([upsample_p3,latlayer_2c], name='add_p2')
        toplayer_p2 = FPN.deconv(add_p2, 4, 4, num_classes, 4, 4, padding=0, relu=False, name='toplayer_p2')

        outputs = toplayer_p2
        print(outputs.name)
        
        return Model(inputs, outputs, teacher, is_training)

class UNet:
    def __init__(self, size=(128, 128), l2_reg=None):
        self.model = self.create_model(size, l2_reg)

    @staticmethod
    def create_model(size, l2_reg):
        inputs = tf.placeholder(tf.float32, [None, size[0], size[1], 3])
        teacher = tf.placeholder(tf.float32, [None, size[0], size[1], len(ld.DataSet.CATEGORY)])
        is_training = tf.placeholder(tf.bool)

        # 1, 1, 3
        conv1_1 = UNet.conv(inputs, filters=64, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv1_2 = UNet.conv(conv1_1, filters=64, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool1 = UNet.pool(conv1_2)

        # 1/2, 1/2, 64
        conv2_1 = UNet.conv(pool1, filters=128, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv2_2 = UNet.conv(conv2_1, filters=128, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool2 = UNet.pool(conv2_2)

        # 1/4, 1/4, 128
        conv3_1 = UNet.conv(pool2, filters=256, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv3_2 = UNet.conv(conv3_1, filters=256, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool3 = UNet.pool(conv3_2)

        # 1/8, 1/8, 256
        conv4_1 = UNet.conv(pool3, filters=512, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv4_2 = UNet.conv(conv4_1, filters=512, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool4 = UNet.pool(conv4_2)

        # 1/16, 1/16, 512
        conv5_1 = UNet.conv(pool4, filters=1024, l2_reg_scale=l2_reg)
        conv5_2 = UNet.conv(conv5_1, filters=1024, l2_reg_scale=l2_reg)
        concated1 = tf.concat([UNet.conv_transpose(conv5_2, filters=512, l2_reg_scale=l2_reg), conv4_2], axis=3)

        conv_up1_1 = UNet.conv(concated1, filters=512, l2_reg_scale=l2_reg)
        conv_up1_2 = UNet.conv(conv_up1_1, filters=512, l2_reg_scale=l2_reg)
        concated2 = tf.concat([UNet.conv_transpose(conv_up1_2, filters=256, l2_reg_scale=l2_reg), conv3_2], axis=3)

        conv_up2_1 = UNet.conv(concated2, filters=256, l2_reg_scale=l2_reg)
        conv_up2_2 = UNet.conv(conv_up2_1, filters=256, l2_reg_scale=l2_reg)
        concated3 = tf.concat([UNet.conv_transpose(conv_up2_2, filters=128, l2_reg_scale=l2_reg), conv2_2], axis=3)

        conv_up3_1 = UNet.conv(concated3, filters=128, l2_reg_scale=l2_reg)
        conv_up3_2 = UNet.conv(conv_up3_1, filters=128, l2_reg_scale=l2_reg)
        concated4 = tf.concat([UNet.conv_transpose(conv_up3_2, filters=64, l2_reg_scale=l2_reg), conv1_2], axis=3)

        conv_up4_1 = UNet.conv(concated4, filters=64, l2_reg_scale=l2_reg)
        conv_up4_2 = UNet.conv(conv_up4_1, filters=64, l2_reg_scale=l2_reg)
        outputs = UNet.conv(conv_up4_2, filters=ld.DataSet.length_category(), kernel_size=[1, 1], activation=None)

        return Model(inputs, outputs, teacher, is_training)

    @staticmethod
    def conv(inputs, filters, kernel_size=[3, 3], activation=tf.nn.relu, l2_reg_scale=None, batchnorm_istraining=None):
        if l2_reg_scale is None:
            regularizer = None
        else:
            regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
        conved = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation=activation,
            kernel_regularizer=regularizer
        )
        if batchnorm_istraining is not None:
            conved = UNet.bn(conved, batchnorm_istraining)

        return conved

    @staticmethod
    def bn(inputs, is_training):
        normalized = tf.layers.batch_normalization(
            inputs=inputs,
            axis=-1,
            momentum=0.9,
            epsilon=0.001,
            center=True,
            scale=True,
            training=is_training,
        )
        return normalized

    @staticmethod
    def pool(inputs):
        pooled = tf.layers.max_pooling2d(inputs=inputs, pool_size=[2, 2], strides=2)
        return pooled

    @staticmethod
    def conv_transpose(inputs, filters, l2_reg_scale=None):
        if l2_reg_scale is None:
            regularizer = None
        else:
            regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
        conved = tf.layers.conv2d_transpose(
            inputs=inputs,
            filters=filters,
            strides=[2, 2],
            kernel_size=[2, 2],
            padding='same',
            activation=tf.nn.relu,
            kernel_regularizer=regularizer
        )
        return conved


class Model:
    def __init__(self, inputs, outputs, teacher, is_training):
        self.inputs = inputs
        self.outputs = outputs
        self.teacher = teacher
        self.is_training = is_training

if __name__ == "__main__":
    FPN.create_model()
