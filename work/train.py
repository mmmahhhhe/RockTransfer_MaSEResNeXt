# -*- coding: UTF-8 -*-
"""
图像分类模型的训练主体
"""
import paddle.fluid as fluid
import numpy as np
import paddle
import reader
import os
import utils
import config
from ma_convcardseresnext import Ma_ConvCardSeResNeXt


def build_optimizer(parameter_list=None):
    """
    构建优化器
    :return:
    """
    epoch = config.train_parameters["num_epochs"]
    batch_size = config.train_parameters["train_batch_size"]
    iters = config.train_parameters["train_image_count"] // batch_size
    learning_strategy = config.train_parameters['sgd_strategy']
    lr = learning_strategy['learning_rate']

    boundaries = [int(epoch * i * iters) for i in learning_strategy["lr_epochs"]]
    values = [i * lr for i in learning_strategy["lr_decay"]]
    # utils.logger.info("use Adam optimizer, learning rate boundaries: {} values: {}".format(boundaries, values))
    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=fluid.layers.piecewise_decay(boundaries, values),
                                             regularization=fluid.regularizer.L2Decay(0.00005),
                                             parameter_list=parameter_list)
    utils.logger.info("use Adam optimizer")
    # optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    return optimizer


def load_params(model, optimizer):
    """
    加载模型参数
    :param model:
    :return:
    """
    if config.train_parameters["continue_train"] and os.path.exists(config.train_parameters['save_model_dir']+'.pdparams'):
        utils.logger.info("load params from {}".format(config.train_parameters['save_model_dir']))
        # params, _ = fluid.dygraph.load_persistables(config.train_parameters['save_model_dir'])
        para_dict, opti_dict = fluid.dygraph.load_dygraph(config.train_parameters['save_model_dir'])
        model.set_dict(para_dict)
    if config.train_parameters["continue_train"] and os.path.exists(config.train_parameters['save_model_dir']+'.pdopt'):
        optimizer.set_dict(opti_dict)
        # model.load_dict(params)


def train():
    """
    训练主体
    :return:
    """
    # 会自动根据当前 paddle 是CPU版本还是GPU版本选择运行硬件
    # 如果是 GPU，默认使用第 0 块
    # 如果希望指定使用，需要主动传入 place 变量，或者通过设置 CUDA_VISIBLE_DEVICES 环境变量控制可见显卡
    utils.logger.info("start train")
    with fluid.dygraph.guard():
        epoch_num = config.train_parameters["num_epochs"]
        # mobilenet = net(1.0, config.train_parameters['class_dim'])
        
        net = Ma_ConvCardSeResNeXt(config.train_parameters['class_dim'])

        optimizer = build_optimizer(parameter_list=net.parameters())
        load_params(net, optimizer)
        file_list = os.path.join(config.train_parameters['data_dir'], config.train_parameters['train_file_list'])
        custom_reader = reader.custom_image_reader(file_list, config.train_parameters['data_dir'], mode='train')
        train_reader = paddle.batch(custom_reader,
                                    batch_size=config.train_parameters['train_batch_size'],
                                    drop_last=True)
        current_acc = 0.0
        to_save_stat_dict = None
        for current_epoch in range(epoch_num):
            epoch_acc = 0.0
            batch_count = 0
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0] for x in data]).astype('float32')
                y_data = np.array([[x[1]] for x in data]).astype('int')

                img = fluid.dygraph.to_variable(dy_x_data)
                label = fluid.dygraph.to_variable(y_data)
                label.stop_gradient = True

                out, acc = net(img, label)
                softmax_out = fluid.layers.softmax(out, use_cudnn=False)
                loss = fluid.layers.cross_entropy(softmax_out, label)
                avg_loss = fluid.layers.mean(loss)

                # 通过这句话求出整个网络，所有参数的梯度
                avg_loss.backward()
                # 在优化器的指导下，每个参数根据自身梯度进行更新
                optimizer.minimize(avg_loss)
                net.clear_gradients()
                batch_count += 1
                epoch_acc += acc.numpy()
                if batch_id % 5 == 0 and batch_id != 0:
                    utils.logger.info("loss at epoch {} step {}: {}, acc: {}"
                                      .format(current_epoch, batch_id, avg_loss.numpy(), acc.numpy()))

            epoch_acc /= batch_count
            utils.logger.info("epoch {} acc: {}".format(current_epoch, epoch_acc))
            if epoch_acc >= current_acc:
                utils.logger.info("current epoch {} acc: {} better than last acc: {}, save model"
                                  .format(current_epoch, epoch_acc, current_acc))
                current_acc = epoch_acc
                
                fluid.dygraph.save_dygraph(net.state_dict(), config.train_parameters['save_model_dir'])
                fluid.dygraph.save_dygraph(optimizer.state_dict(), config.train_parameters['save_model_dir'])
        utils.logger.info("train till end")
        # for k, v in to_save_stat_dict.items():
        #     utils.logger.info("key:{}   value:{}".format(k, v.numpy()))


if __name__ == "__main__":
    train()
