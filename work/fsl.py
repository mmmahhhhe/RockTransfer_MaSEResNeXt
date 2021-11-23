import paddlefsl
import paddle
import os, random
import reader
from ma_convcardseresnext import *
from paddlefsl.model_zoo import maml
from paddlefsl.datasets import MiniImageNet
from reader import *
from paddlefsl.task_sampler import TaskSet
import paddle.nn.functional as F

%load_ext autoreload
%autoreload 2

from res18 import *


class FSLDataset(paddlefsl.datasets.CVDataset):
    def __init__(self, file_list, mode='train'):
        self.mode = mode
        self.class_dict  = {}

        with codecs.open(file_list) as flist:
            self.lines = [line.strip() for line in flist]
            
        # np.random.shuffle(self.lines)
        self.class_dict = {}
        for num, line in enumerate(self.lines):
            img_path, label = line.split()
            self.class_dict[label] = self.class_dict.get(label, []) + [num]



    def __getitem__(self, idx):

        line = self.lines[idx]
        if self.mode == 'train':
            img_path, label = line.split()
            img = Image.open(img_path)
            img = resize_img(img, train_parameters['input_size'])

            if img.mode != 'RGB':
                img = img.convert('RGB')
            if train_parameters['image_enhance_strategy']['need_distort']:
                img = distort_color(img)
            if train_parameters['image_enhance_strategy']['need_rotate']:
                img = rotate_image(img)
            if train_parameters['image_enhance_strategy']['need_crop']:
                img = random_crop(img, train_parameters['input_size'])
            if train_parameters['image_enhance_strategy']['need_flip']:
                mirror = int(np.random.uniform(0, 2))
                if mirror == 1:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # HWC--->CHW && normalized
            img = np.array(img).astype('float32')
            img -= train_parameters['mean_rgb']
            img = img.transpose((2, 0, 1))  # HWC to CHW
            img *= 0.007843 
            return img, int(label)

        else:
            img_path, label = line.split()
            img = Image.open(img_path)

            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = resize_img(img, train_parameters['input_size'])
            # HWC--->CHW && normalized
            img = np.array(img).astype('float32')
            img -= train_parameters['mean_rgb']
            img = img.transpose((2, 0, 1))  # HWC to CHW
            img *= 0.007843
            return img, int(label)
            
    def __len__(self):
        return len(self.lines)

    def sample_task_set(self, ways=5, shots=5, query_num=1):
        query_num = shots if query_num is None else query_num
        sample = random.sample(list(self.class_dict.keys()), ways)
        # result: List[ (str(label name), List[np.ndarray(image)]) ]
        result = []
        for label in sample:
            image_list = [self.__getitem__(i)[0] for i in random.sample(self.class_dict[label], shots + query_num)]
            result.append((label, image_list))
        return TaskSet(label_names_data=result, ways=ways, shots=shots, query_num=query_num)


file_list = 'case3/dataset/train.txt'
training_set = FSLDataset(file_list, mode='train')
valid_set = FSLDataset(file_list, mode='eval')

net = paddlefsl.backbones.Conv(input_size=(3, 224, 224), output_size=4)
train_dir = maml.meta_training(train_dataset=training_set,
                                valid_dataset=valid_set,
                                ways=4,
                                shots=1,
                                model=net,
                                
                                meta_lr=2e-1,
                                inner_lr=2e-1,
                                iterations=10000,
                                meta_batch_size=32,
                                inner_adapt_steps=5,
                                report_iter=10)
paddle.save(net.state_dict(), "fsl/fsl_net.pdparams")
# net_1 = paddlefsl.backbones.Conv(input_size=(3, 224, 224), output_size=4)
# state_dict = paddle.load("fsl/fsl_net.pdparams")
# net_1.set_state_dict(state_dict)


main_net = Ma_ConvCardSeResNeXt(4)

thes = 1e-1
file_list = 'case3/dataset/eval.txt'
custom_reader = reader.custom_image_reader(file_list, '.', mode='val')
custom_reader = paddle.batch(custom_reader,
                            batch_size=15,
                            drop_last=False)


for batch_id, data in enumerate(custom_reader()):
    dy_x_data = np.array([x[0] for x in data]).astype('float32')
    y_data = np.array([[x[1]] for x in data]).astype('int')

    img = fluid.dygraph.to_variable(dy_x_data)
    label = fluid.dygraph.to_variable(y_data)
    label.stop_gradient = True

    out, acc = main_net(img, label)
    print(acc)
    fsl_out = net(img)
    fsl_acc = fluid.layers.accuracy(input=fsl_out, label=label)
    mix_out = (1 - thes) * out + fsl_out * thes
    mix_acc = fluid.layers.accuracy(input=mix_out, label=label)

print(mix_acc)









