import codecs
import os
import utils

train_parameters = {  
    "data_dir": ".",  # 训练数据存储地址  
    "num_epochs": 10000,  
    "train_batch_size": 64, 
    "infer_img": 'img.jpg',


    "mean_rgb": [85, 96, 102],  # 常用图片的三通道均值，通常来说需要先对训练数据做统计，此处仅取中间值  
    
    "input_size": [3, 224, 224],  
    "class_dim": -1,  # 分类数，会在初始化自定义 reader 的时候获得  
    "image_count": -1,  # 训练图片数量，会在初始化自定义 reader 的时候获得  
    "label_dict": {},  
    "train_file_list": "train.txt",  
    "eval_file_list": "eval.txt",
    "label_file": "label_list.txt",  
    "save_model_dir": "./save_dir/model",  
    "continue_train": True,        # 是否接着上一次保存的参数接着训练，优先级高于预训练模型  
    "image_enhance_strategy": {  # 图像增强相关策略  
        "need_distort": True,  # 是否启用图像颜色增强  
        "need_rotate": True,   # 是否需要增加随机角度  
        "need_crop": True,      # 是否要增加裁剪  
        "need_flip": True,      # 是否要增加水平随机翻转  
        "hue_prob": 0.5,  
        "hue_delta": 18,  
        "contrast_prob": 0.5,  
        "contrast_delta": 0.5,  
        "saturation_prob": 0.5,  
        "saturation_delta": 0.5,  
        "brightness_prob": 0.5,  
        "brightness_delta": 0.125  
    },  
    "early_stop": {  
        "sample_frequency": 50,  
        "successive_limit": 3,  
        "good_acc1": 0.92  
    },  
    "rsm_strategy": {  
        "learning_rate": 0.001,  
        "lr_epochs": [20, 40, 60, 80, 100],  
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]  
    },  
    "momentum_strategy": {  
        "learning_rate": 0.001,  
        "lr_epochs": [20, 40, 60, 80, 100],  
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]  
    },  
    "sgd_strategy": {  
        "learning_rate": 0.001,  
        "lr_epochs": [20, 40, 60, 80, 100],  
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]  
    },  
    "adam_strategy": {  
        "learning_rate": 0.002  
    }  
}  

def init_train_parameters():
    """
    初始化训练参数，主要是初始化图片数量，类别数
    :return:
    """

    label_list = os.path.join(train_parameters['data_dir'], train_parameters['label_file'])
    index = 0
    with codecs.open(label_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        for line in lines:
            parts = line.strip().split()
            train_parameters['label_dict'][parts[1]] = int(parts[0])
            index += 1
        train_parameters['class_dim'] = index

    train_file_list = os.path.join(train_parameters['data_dir'], train_parameters['train_file_list'])
    with codecs.open(train_file_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        train_parameters['train_image_count'] = len(lines)

    eval_file_list = os.path.join(train_parameters['data_dir'], train_parameters['eval_file_list'])
    with codecs.open(eval_file_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        train_parameters['eval_image_count'] = len(lines)

    utils.logger.info("input_size: {}".format(train_parameters['input_size']))
    utils.logger.info("class_dim: {}".format(train_parameters['class_dim']))
    utils.logger.info("continue_train: {}".format(train_parameters['continue_train']))
    utils.logger.info("train_image_count: {}".format(train_parameters['train_image_count']))
    utils.logger.info("eval_image_count: {}".format(train_parameters['eval_image_count']))
    utils.logger.info("num_epochs: {}".format(train_parameters['num_epochs']))
    utils.logger.info("train_batch_size: {}".format(train_parameters['train_batch_size']))
    utils.logger.info("mean_rgb: {}".format(train_parameters['mean_rgb']))
    utils.logger.info("save_model_dir: {}".format(train_parameters['save_model_dir']))


init_train_parameters()