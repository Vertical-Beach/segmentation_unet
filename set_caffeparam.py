import os
import tensorflow as tf
import numpy as np
from util import model

param_1 = ['upsample_p5','upsample_p4','upsample_p3']
param_2 = ['conv1_7x7_s2','conv2_3x3_reduce','conv2_3x3','inception_3a_1x1','inception_3a_3x3_reduce','inception_3a_3x3','inception_3a_5x5_reduce','inception_3a_5x5','inception_3a_pool_proj','inception_3b_1x1','inception_3b_3x3_reduce','inception_3b_3x3','inception_3b_5x5_reduce','inception_3b_5x5','inception_3b_pool_proj','inception_4a_1x1','inception_4a_3x3_reduce','inception_4a_3x3','inception_4a_5x5_reduce','inception_4a_5x5','inception_4a_pool_proj','inception_4b_1x1','inception_4b_3x3_reduce','inception_4b_3x3','inception_4b_5x5_reduce','inception_4b_5x5','inception_4b_pool_proj','inception_4c_1x1','inception_4c_3x3_reduce','inception_4c_3x3','inception_4c_5x5_reduce','inception_4c_5x5','inception_4c_pool_proj','inception_4d_1x1','inception_4d_3x3_reduce','inception_4d_3x3','inception_4d_5x5_reduce','inception_4d_5x5','inception_4d_pool_proj','inception_4e_1x1','inception_4e_3x3_reduce','inception_4e_3x3','inception_4e_5x5_reduce','inception_4e_5x5','inception_4e_pool_proj','inception_5a_1x1','inception_5a_3x3_reduce','inception_5a_3x3','inception_5a_5x5_reduce','inception_5a_5x5','inception_5a_pool_proj','inception_5b_1x1','inception_5b_3x3_reduce','inception_5b_3x3','inception_5b_5x5_reduce','inception_5b_5x5','inception_5b_pool_proj','p5','latlayer_4f','toplayer_p4','latlayer_3d','toplayer_p3','latlayer_2c','toplayer_p2']
param_5 = ['conv1_7x7_s2_BatchNorm','conv2_3x3_reduce_BatchNorm','conv2_3x3_BatchNorm','inception_3a_1x1_BatchNorm','inception_3a_3x3_reduce_BatchNorm','inception_3a_3x3_BatchNorm','inception_3a_5x5_reduce_BatchNorm','inception_3a_5x5_BatchNorm','inception_3a_pool_proj_BatchNorm','inception_3b_1x1_BatchNorm','inception_3b_3x3_reduce_BatchNorm','inception_3b_3x3_BatchNorm','inception_3b_5x5_reduce_BatchNorm','inception_3b_5x5_BatchNorm','inception_3b_pool_proj_BatchNorm','inception_4a_1x1_BatchNorm','inception_4a_3x3_reduce_BatchNorm','inception_4a_3x3_BatchNorm','inception_4a_5x5_reduce_BatchNorm','inception_4a_5x5_BatchNorm','inception_4a_pool_proj_BatchNorm','inception_4b_1x1_BatchNorm','inception_4b_3x3_reduce_BatchNorm','inception_4b_3x3_BatchNorm','inception_4b_5x5_reduce_BatchNorm','inception_4b_5x5_BatchNorm','inception_4b_pool_proj_BatchNorm','inception_4c_1x1_BatchNorm','inception_4c_3x3_reduce_BatchNorm','inception_4c_3x3_BatchNorm','inception_4c_5x5_reduce_BatchNorm','inception_4c_5x5_BatchNorm','inception_4c_pool_proj_BatchNorm','inception_4d_1x1_BatchNorm','inception_4d_3x3_reduce_BatchNorm','inception_4d_3x3_BatchNorm','inception_4d_5x5_reduce_BatchNorm','inception_4d_5x5_BatchNorm','inception_4d_pool_proj_BatchNorm','inception_4e_1x1_BatchNorm','inception_4e_3x3_reduce_BatchNorm','inception_4e_3x3_BatchNorm','inception_4e_5x5_reduce_BatchNorm','inception_4e_5x5_BatchNorm','inception_4e_pool_proj_BatchNorm','inception_5a_1x1_BatchNorm','inception_5a_3x3_reduce_BatchNorm','inception_5a_3x3_BatchNorm','inception_5a_5x5_reduce_BatchNorm','inception_5a_5x5_BatchNorm','inception_5a_pool_proj_BatchNorm','inception_5b_1x1_BatchNorm','inception_5b_3x3_reduce_BatchNorm','inception_5b_3x3_BatchNorm','inception_5b_5x5_reduce_BatchNorm','inception_5b_5x5_BatchNorm','inception_5b_pool_proj_BatchNorm']
filename_dic = {}
for param in param_1:
    filename_dic[param + '/kernel'] = param + '_0.npy'
for param in param_2:
    filename_dic[param + '/kernel'] = param + '_0.npy'
    filename_dic[param + '/bias'] = param + '_1.npy'
for param in param_5:
    filename_dic[param + '/gamma'] = param + '_0.npy'
    filename_dic[param + '/beta'] = param + '_1.npy'
    filename_dic[param + '/moving_mean'] = param + '_2.npy'
    filename_dic[param + '/moving_variance'] = param + '_3.npy'
not_converted_yet_list = list(filename_dic.keys())

with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False, name='global_step')
    images = [tf.placeholder(tf.float32, [1, 480, 960, 3])]
    labels = [tf.placeholder(tf.int32, [1, 480, 960, 1])]

    # Build model
    # print("Build ResNet-18 model")
    # hp = resnet.HParams(batch_size=2,
    #                     num_gpus=1,
    #                     num_classes=1000,
    #                     weight_decay=0.001,
    #                     momentum=0.9,
                        
    #                     finetune=False)
    # network_train = resnet.ResNet(hp, images, labels, global_step, name="train")
    # network_train.build_model()
    model = model.FPN().model
    # print('Number of Weights: %d' % model._weights)
    # print('FLOPs: %d' % model._flops)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.96),
        allow_soft_placement=True,
        log_device_placement=False))
    sess.run(init)

    # Set variables values
    print('Set variables to loaded weights')
    all_vars = tf.global_variables()
    for v in all_vars:
        if v.op.name == 'global_step':
            continue
        print('\t' + v.op.name)
        if v.op.name not in filename_dic:
            raise Exception(v.op.name + " does not exist in filename_dic")
        filepath = '/workspace/Vitis-AI-Tutorials/files/Segment/workspace/model/extract-caffe-params/hoge/' + filename_dic[v.op.name]
        if os.path.exists(filepath) is False:
            raise Exception(filepath)
        weight = np.load(filepath)
        if v.op.name.endswith('kernel'):
            #NCHW -> HWCN
            weight = np.transpose(weight, [2,3,1,0])
        elif any([v.op.name.endswith(n) for n in ['gamma', 'beta', 'moving_mean', 'moving_variance']]):
            weight = np.reshape(weight, -1)
        print(weight.shape)
        assign_op = v.assign(weight)
        sess.run(assign_op)
        not_converted_yet_list.remove(v.op.name)

    # check all weights are converted
    if len(not_converted_yet_list) != 0:
        print('WARNING* not_converted_yet_list: ', not_converted_yet_list)
    else:
        print('All weights are converted successfully!!')
    # Save as checkpoint
    INIT_CHECKPOINT_DIR = 'converted_weight'
    os.makedirs(INIT_CHECKPOINT_DIR, exist_ok=True)
    print('Save as checkpoint: %s' % INIT_CHECKPOINT_DIR)
    if not os.path.exists(INIT_CHECKPOINT_DIR):
        os.mkdir(INIT_CHECKPOINT_DIR)
    saver = tf.train.Saver(tf.global_variables())
    saver.save(sess, os.path.join(INIT_CHECKPOINT_DIR, 'model.ckpt'))

print('Done!')