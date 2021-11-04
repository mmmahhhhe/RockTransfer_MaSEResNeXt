import paddle.fluid as fluid
import numpy as np
import paddle
import reader
import os
import utils
import config
from ma_convcardseresnext import Ma_ConvCardSeResNeXt


from ma_convcardseresnext import *
class Master(fluid.dygraph.Layer):
    def __init__(self, class_dim=105):
        super(Master, self).__init__()


        cardinality = 128
        reduction_ratio = 16
        depth = [3, 8, 16, 3]
        num_filters = [128, 256, 512, 1024]
        self.conv0 = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=5,
            stride=2,
            act='relu')
        self.pool = Pool2D(
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='avg')

        self.bottleneck_block_list = []
        num_channels = 64

        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        cardinality=cardinality,
                        reduction_ratio=reduction_ratio,
                        shortcut=shortcut))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        self.pool2d_avg = Pool2D(
            pool_size=7, pool_type='max', global_pooling=True)
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.pool2d_avg_output = num_filters[len(num_filters) - 1] * 2 * 1 * 1

    def forward(self, inputs, label=None):
        y = self.conv0(inputs)
        y = self.pool(y)


        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
        y = fluid.layers.dropout(y, dropout_prob=0.5, seed=100)
        y = fluid.layers.reshape(y, shape=[-1, self.pool2d_avg_output])
        return y


class Transfer(fluid.dygraph.layers.Layer):

    def __init__(self, net, out_class):
        super(Transfer, self).__init__()
        self.net = net
        self.out = fluid.dygraph.Linear(2048, out_class, act='softmax',
                             param_attr=fluid.param_attr.ParamAttr(name="linear_transfer_1.w_0"),
                             bias_attr=fluid.param_attr.ParamAttr(name="linear_transfer_1.b_0")
                             )
    def forward(self, x, label=None):
        x = self.net(x)
        x.stop_gradient = True
        y = self.out(x)
        if label is not None:
            acc = fluid.layers.accuracy(input=y, label=label)
            return y, acc        
        else:
            return y       


def model_dict(model):
    
    trans_dict = {}
    master_dict = model.state_dict()
    for param in master_dict:
        tran_name = param
        if param[:4] == 'net.':
            tran_name = param[4:]
        trans_dict[tran_name] = master_dict[param]
    # trans_dict = model_dict(model=transfer)
    return trans_dict


# 迁移
trans_fc_epoch = 5

with fluid.dygraph.guard():
    master_model = 'master_model'

    # 迁移模型
    master = Master()
    para_dict, opti_dict = fluid.dygraph.load_dygraph(master_model + '/model')
    for i in list(para_dict)[-2:]:
        para_dict.pop(i)
    
    master.set_dict(para_dict)

    transfer = Transfer(master, config.train_parameters['class_dim'])
    optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.2, parameter_list=transfer.parameters())

    # 数据迭代器
    file_list = os.path.join(config.train_parameters['data_dir'], config.train_parameters['train_file_list'])
    custom_reader = reader.custom_image_reader(file_list, config.train_parameters['data_dir'], mode='train')
    train_reader = paddle.batch(custom_reader,
                                batch_size=config.train_parameters['train_batch_size'],
                                drop_last=True)

    for current_epoch in range(trans_fc_epoch):
        epoch_acc = 0.0
        batch_count = 0
        for batch_id, data in enumerate(train_reader()):
            dy_x_data = np.array([x[0] for x in data]).astype('float32')
            y_data = np.array([[x[1]] for x in data]).astype('int')
            img = fluid.dygraph.to_variable(dy_x_data)
            label = fluid.dygraph.to_variable(y_data)
            label.stop_gradient = True

            out, acc = transfer(img, label)
            softmax_out = fluid.layers.softmax(out, use_cudnn=False)
            loss = fluid.layers.cross_entropy(softmax_out, label)
            avg_loss = fluid.layers.mean(loss)

            # 通过这句话求出整个网络，所有参数的梯度
            avg_loss.backward()
            # 在优化器的指导下，每个参数根据自身梯度进行更新
            optimizer.minimize(avg_loss)
            transfer.clear_gradients()

            batch_count += 1
            epoch_acc += acc.numpy()
            if batch_id % 5 == 0:
                utils.logger.info("loss at epoch {} step {}: {}, acc: {}"
                                    .format(current_epoch, batch_id, avg_loss.numpy(), acc.numpy()))
        
        epoch_acc /= batch_count

        trans_dict = model_dict(model=transfer)
        fluid.dygraph.save_dygraph(trans_dict, config.train_parameters['save_model_dir'])

        utils.logger.info("epoch {} acc: {}".format(current_epoch, epoch_acc))
    utils.logger.info("Transfer till end")
