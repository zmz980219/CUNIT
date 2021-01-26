# this py file is used to preprocess data to get information we need
# input: img with end '.jpg'
# output: 1) '.jpg' file which is valid img
#         2) '.npy' file which contains bounding box of detected people in numpy array
#               box format: [left, top, right, bottom, resizex, resizey]
#         3) '.png' file which contains the cloth mask, in the experiment we choose upper cloth
# to be mentioned, 1) and 2) are in the same folder 'images', 3) is in folder 'masks'
# p.s. cloth mask file are made from resized img which we choose 256 * 256 resolution


import os
import torch
import argparse
import networks
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import cv2
from data.networks import init_model, dataset_settings
from data.utils.transforms import get_affine_transform, _box2cs, transform_logits
from tqdm import tqdm


def main(args):
    # load imgs from path
    img_path = args.dataroot
    # check if exists img_path
    assert os.path.exists(img_path), 'illegal path!'
    save_path = args.saveroot
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    folder2 = os.path.join(save_path, 'masks')
    if not os.path.exists(folder2):
        os.mkdir(folder2)
    print('Loading imgs from {} ...'.format(img_path))
    imgs_list = [f for f in os.listdir(img_path) if f.endswith(".jpg")]
    print('Get imgs total number:', len(imgs_list))
    # object detector for person
    print('Loading models...')
    # human segmentation for upper cloth
    num_classes = dataset_settings[args.seg_dataset]['num_classes']
    seg_model = init_model('resnet101', num_classes=num_classes, pretrained=None)
    state_dict = torch.load(args.seg_model_restore)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    seg_model.load_state_dict(new_state_dict)
    seg_model.eval()
    print('Models done!')

    device = torch.device('cuda:{}'.format(args.gpu) if args.gpu != -1 else torch.device('cpu'))
    seg_model.to(device)
    instanceTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    for img in tqdm(imgs_list):
        # load img
        real_img_path = os.path.join(img_path, img)
        # print('image size', image.size())
        person_cvImg = cv2.imread(real_img_path, cv2.IMREAD_COLOR)
        input_size = list()
        input_size.append(args.resize)
        input_size.append(args.resize)

        input = cv2.resize(person_cvImg, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR)
        h, w, c = input.shape
        person_center, s = _box2cs([0, 0, w - 1, h - 1])
        # print('shape: ', h, w)
        # cv2.imshow('image', input)
        # cv2.waitKey(0)
        input = instanceTransform(input)
        input = input.unsqueeze(0)
        # print('input size of seg:', input.size())
        with torch.no_grad():
            output = seg_model(input.to(device))
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC
            logits_result = transform_logits(upsample_output.data.cpu().numpy(), person_center, s, w, h,
                                             input_size=input_size)
            parsing_result = np.argmax(logits_result, axis=2)
            output_arr = np.asarray(parsing_result, dtype=np.uint8)
            output_img = Image.fromarray(np.asarray(output_arr, dtype=np.uint8))
            save_mask_path = os.path.join(folder2, str(img.split('.')[0] + '.png'))
            output_img.save(save_mask_path)

    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--saveroot', type=str, required=True)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--seg_dataset', type=str, default='atr')
    parser.add_argument('--seg_model_restore', type=str, default='data/exp-schp-201908301523-atr.pth')
    parser.add_argument('--gpu', type=int, default=0, help='-1 for cpu')
    # parser.add_argument()
    args = parser.parse_args()
    main(args)

