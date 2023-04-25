import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, Resizer, Normalizer
from retinanet import coco_eval
from collections import OrderedDict

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--model_path', help='Path to model', type=str)

    parser = parser.parse_args(args)

    dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                              transform=transforms.Compose([Normalizer(), Resizer()]))

    # Create the model
    # retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    retinanet = torch.load(parser.model_path)    # 直接加载训练好模型

    # # 获得模型的原始状态以及参数。
    # params = retinanet.state_dict()
    # for k, v in params.items():
    #     print(k)  # 只打印key值，不打印具体参数。

    # # 这个方法能够直接打印出保存的checkpoint的键和值。
    # checkpoint = torch.load(parser.model_path)
    # # Load weights to resume from checkpoint。
    # for k, v in checkpoint.items():
    #     print(k)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():

        # weights = torch.load(parser.model_path)    #首先通过 torch.load() 加载权重文件，然后遍历字典，如果 key 中包含 'module' 则将其删掉
        # weights_dict = {}
        # for k, v in weights.items():
        #     new_k = k.replace('module.', '') if 'module' in k else k
        #     weights_dict[new_k] = v
        # retinanet.load_state_dict(weights_dict)
        #
        # pre_dict = torch.load(parser.model_path)
        # new_pre = {}
        # for k, v in pre_dict.items():
        #     name = k[7:]
        #     new_pre[name] = v
        # retinanet.load_state_dict(new_pre)
        #
        # original saved file with DataParallel
        # create new OrderedDict that does not contain `module.`
        # state_dict = torch.load(parser.model_path)
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        #     new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
        # # load params
        # retinanet.load_state_dict(new_state_dict)  # 从新加载这个模型。

        # retinanet.load_state_dict(torch.load(parser.model_path), strict= False)  # 加载训练好的权重参数
        retinanet = torch.nn.DataParallel(retinanet).cuda()

    else:
        retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    coco_eval.evaluate_coco(dataset_val, retinanet)


if __name__ == '__main__':
    main()
