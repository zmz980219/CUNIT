# 0124 new version
# remove ClothTransfer and DisClothTransfer and losses involved
# add a new part to transfer content information, the content information mentioned is concatenated by
# image content and mask content
# add new loss function to guide the running of content transfer
# TODO: keep the original scale? 400, 600 -> 200, 300
# TODO: fuse content encoder into ones? and cancel the content translator(cause we also have resblock in encoder
import torch
import torchvision
import os
from options import TrainOptions
from dataset import dataset_unpair
from model import CUNIT
from torchvision import transforms
from tqdm import tqdm
from saver import Saver

# tensorboard --logdir logs/#name# --bind_all
def main():
    # parse options
    parser = TrainOptions()
    opts = parser.parse()

    # data loader
    print('\n--- load dataset ---')
    dataset = dataset_unpair(opts)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True,
                                               num_workers=opts.nThreads)

    # model
    print('\n--- load model ---')
    model = CUNIT(opts)
    model.setgpu(opts.gpu)
    if opts.resume is None:
        model.initialize()
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(opts.resume)
    model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    print('start the training at epoch %d' % (ep0))
    # saver for display and output
    saver = Saver(opts)

    # train
    print('\n--- train ---')
    # print(model)
    max_it = 500000
    # considering the situation that bugs come out with two modules
    # working together, perhaps separating training is a better way
    for ep in range(ep0, opts.n_ep):
        for it, data in tqdm(enumerate(train_loader)):
            # 'A': A, 'B': B, 'A_mask': A_mask, 'B_mask': B_mask, 'A_box': A_box, 'B_box': B_box
            images_a = data['A']
            images_b = data['B']
            masks_a = data['A_mask']
            masks_b = data['B_mask']
            # boxes_a = data['A_box']
            # boxes_b = data['B_box']
            # person_a = data['A_crop']
            # person_b = data['B_crop']
            if images_a.size(0) != opts.batch_size or images_b.size(0) != opts.batch_size:
                continue

            # input data
            images_a = images_a.cuda(opts.gpu).detach()
            images_b = images_b.cuda(opts.gpu).detach()
            masks_a = masks_a.cuda(opts.gpu).detach()
            masks_b = masks_b.cuda(opts.gpu).detach()
            # person_a = person_a.cuda(opts.gpu).detach()
            # person_b = person_b.cuda(opts.gpu).detach()
            model.update(images_a, images_b, masks_a, masks_b)

            if not opts.no_display_img:
                saver.write_display(total_it, model)

            total_it = total_it + 1
            if total_it >= max_it:
                break

        # decay learning rate
        if opts.n_ep_decay > -1:
            model.update_lr()

        saver.write_img(ep, model)
        saver.write_model(ep, total_it, model)



def save_images(opts, ep, model):
    if (ep+1) % opts.img_save_freq == 0:
        images = model.assemble_outputs()
        model_dir = os.path.join(opts.result_dir, opts.name)
        image_dir = os.path.join(model_dir, 'images')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        img_filename = '%s/gen_%05d.jpg' % (image_dir, ep)
        torchvision.utils.save_image(images / 2 + 0.5, img_filename, nrow=1)


if __name__ == '__main__':
    main()

    # # print('images_a size', images_a.size())
    # content_a, content_b = model.EncContent(images_a, images_b)
    # input_content = torch.cat((content_a, content_b), dim=0)
    # print('content_a size:', content_a.size())
    # style_a, style_b = model.EncStyle(images_a, images_b)
    # input_style = torch.cat((style_a, style_b), dim=0)
    # print('input content size', input_content.size())
    # print('input style size', input_style.size())
    # print('style_a size:', style_a.size())
    # res = model.GloResBlock.forward_x(input_content, input_style)
    # print('result size:', res.size())
    # res = model.Gen.forward_x(res)
    # print('new result size:', res.size())
    # image_a1, image_a2, image_b1, image_b2 = res
    # print(image_a1.size())
    # image = unloader(image_a1)
    # image.save('a.jpg')
