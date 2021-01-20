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
    folder1 = os.path.join(save_path, 'images')
    folder2 = os.path.join(save_path, 'masks')
    if not os.path.exists(folder1):
        os.mkdir(folder1)
    if not os.path.exists(folder2):
        os.mkdir(folder2)
    print('Loading imgs from {} ...'.format(img_path))
    imgs_list = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]
    print('Get imgs total number:', len(imgs_list))
    # object detector for person
    print('Loading models...')
    detector = networks.Detector()
    detector.eval()
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
    detector.to(device)
    seg_model.to(device)
    imageTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    instanceTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    for img in tqdm(imgs_list):
        # load img
        real_img_path = os.path.join(img_path, img)
        rawImage = Image.open(real_img_path).convert('RGB')
        resizeImage = rawImage.resize((args.resize, args.resize), Image.BICUBIC)
        resizeImage.save('resize.jpg')
        image = imageTransform(resizeImage)
        image = image.unsqueeze(0)
        # print('image size', image.size())
        with torch.no_grad():
            boxes = detector(image.to(device))
        # we only process one image at one time
        boxes = boxes[0]
        # we only need one person in pics
        if len(boxes) == 1:
            box = boxes[0].cpu().numpy()
            box = [int(i) for i in box]
            # region is a picture and need convert and transform to further process it
            os.system("cp {} {}".format(real_img_path, os.path.join(folder1, img)))

            box.append(args.resize)
            box.append(args.resize)
            # print('box:', box)
            # print('region shape', region.shape)
            # save images and box at folder1
            box = np.array(box)
            box_filename = os.path.join(folder1, img.split('.')[0])
            np.save(box_filename, box)

            # get cloth mask, we use SCHP to do segmentation and the part we need is 5
            # from https://github.com/PeikeLi/Self-Correction-Human-Parsing we know that:
            # 0 'Background', 1 'Hat', 2 'Hair', 3 'Glove', 4 'Sunglasses', 5 'Upper-clothes', 6 'Dress', 7 'Coat',
            # 8 'Socks', 9 'Pants', 10 'Jumpsuits', 11 'Scarf', 12 'Skirt', 13 'Face', 14 'Left-arm', 15 'Right-arm',
            # 16 'Left-leg', 17 'Right-leg', 18 'Left-shoe', 19 'Right-shoe'
            # for jpg it needs COLOR_RGB2BGR / for png it needs COLOR_RGBA2BGRA
            person_cvImg = cv2.imread(real_img_path, cv2.IMREAD_COLOR)
            input_size = list()
            input_size.append(args.resize)
            input_size.append(args.resize)
            # cv2.imshow('image', person_cvImg)
            # h, w, _ = person_cvImg.shape
            # print('shape: {} {}'.format(h, w))

            # center is [w/2, h/2] s is np.array([w, h])
            # person_center, s = _box2cs([0, 0, w-1, h-1])
            # r = 0
            # print('input size:', input_size)
            # trans = get_affine_transform(person_center, s, r, input_size)
            # input = cv2.warpAffine(
            #     person_cvImg,
            #     trans,
            #     (int(input_size[1]), int(input_size[0])),
            #     flags=cv2.INTER_LINEAR,
            #     borderMode=cv2.BORDER_CONSTANT,
            #     borderValue=(0, 0, 0))


            input = cv2.resize(person_cvImg, (input_size[1], input_size[0]), interpolation = cv2.INTER_LINEAR)
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
                logits_result = transform_logits(upsample_output.data.cpu().numpy(), person_center, s, w, h, input_size=input_size)
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
    parser.add_argument('--seg_dataset', type=str, default='lip')
    parser.add_argument('--seg_model_restore', type=str, default='data/exp-schp-201908261155-lip.pth')
    parser.add_argument('--gpu', type=int, default=0, help='-1 for cpu')
    # parser.add_argument()
    args = parser.parse_args()
    main(args)

