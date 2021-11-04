# -*- coding: UTF-8 -*-
"""
加载模型验证
"""
import utils
import paddle.fluid as fluid
import paddle
import reader
from ma_convcardseresnext import Ma_ConvCardSeResNeXt
import numpy as np
import os
import config
import time

def eval_model():
    utils.logger.info("start eval")
    file_list = os.path.join(config.train_parameters['data_dir'], config.train_parameters['eval_file_list'])

    with fluid.dygraph.guard():
        # params, _ = fluid.dygraph.load_persistables(config.train_parameters['save_model_dir'])
        params, _ = fluid.load_dygraph(config.train_parameters['save_model_dir'])
        
        net = Ma_ConvCardSeResNeXt(config.train_parameters['class_dim'])

        net.set_dict(params)
        net.eval()
        test_reader = paddle.batch(reader.custom_image_reader(file_list, reader.train_parameters['data_dir'], 'val'),
                                   batch_size=1,
                                   drop_last=True)
        accs = []
        start_time = time.time()
        for batch_id, data in enumerate(test_reader()):
            dy_x_data = np.array([x[0] for x in data]).astype('float32')
            y_data = np.array([[x[1]] for x in data]).astype('int')

            img = fluid.dygraph.to_variable(dy_x_data)
            label = fluid.dygraph.to_variable(y_data)
            label.stop_gradient = True

            out, acc = net(img, label)
            accs.append(acc.numpy()[0])
        # for k, v in mobilenet.state_dict().items():
        #     utils.logger.info("key:{}   value:{}".format(k, v.numpy()))
    utils.logger.info("test count: {} , acc: {} cost time: {}"
                      .format(config.train_parameters['eval_image_count'], np.mean(accs), time.time() - start_time))


if __name__ == "__main__":
    eval_model()
