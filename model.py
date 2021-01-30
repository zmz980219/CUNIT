import networks
import torch
import torch.nn as nn

import torch.nn.functional as F

# 目前先不使用multi-scale discriminators
class CUNIT(nn.Module):
    def __init__(self, opts):
        super(CUNIT, self).__init__()

        # parameters
        lr = 0.0001
        lr_dcontent = lr / 2.5
        self.no_ms = opts.no_ms
        # styleSpace means the size of style space
        self.styleSpace = 8
        self.size = opts.crop_size

        # discriminators 包括区分图片是否是生成的Dx, Dy
        # 区分content属于哪一类的Dc 和 Dic(这两者损失函数不同)
        self.DisX = networks.Discriminator(opts.input_dim_x, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
        self.DisY = networks.Discriminator(opts.input_dim_y, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
        self.DisContent = networks.ContentDiscriminator()
        self.DisMaskContentX = networks.MaskClassifier(input_nc=1, ndf=64)
        self.DisMaskContentY = networks.MaskClassifier(input_nc=1, ndf=64)

        # encoders 包括style encoders, content encoders
        # 目前采用的是INIT中的实现方式，即Exc和Exci共用一个encoder
        self.EncContent = networks.ContentEncoder(opts.input_dim_x, opts.input_dim_y)
        self.EncMaskContent = networks.ContentEncoder(1, 1, dim=16)
        self.DecMaskContent = networks.ContentMaskDecoder(256, 256, 1, 1)
        self.EncStyle = networks.StyleEncoder(opts.input_dim_x, opts.input_dim_y, self.styleSpace)
        self.TransContent = networks.ContentTranslator(dim=256+64, n_blocks=9)

        # generator
        self.Gen = networks.Generator(opts.input_dim_x, opts.input_dim_y, nz=self.styleSpace)

        # optimizers
        self.DisX_opt = torch.optim.Adam([{'params': self.DisX.parameters(), 'initial_lr': lr}], lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.DisY_opt = torch.optim.Adam([{'params': self.DisY.parameters(), 'initial_lr': lr}], lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.DisContent_opt = torch.optim.Adam([{'params': self.DisContent.parameters(), 'initial_lr': lr}], lr=lr_dcontent, betas=(0.5, 0.999), weight_decay=0.0001)
        self.DisMaskContentX_opt = torch.optim.Adam([{'params': self.DisMaskContentX.parameters(), 'initial_lr': lr}], lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.DisMaskContentY_opt = torch.optim.Adam([{'params': self.DisMaskContentY.parameters(), 'initial_lr': lr}],
                                                    lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.EncContent_opt = torch.optim.Adam([{'params': self.EncContent.parameters(), 'initial_lr': lr}], lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.EncMaskContent_opt = torch.optim.Adam([{'params': self.EncMaskContent.parameters(), 'initial_lr': lr}], lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.DecMaskContent_opt = torch.optim.Adam([{'params': self.DecMaskContent.parameters(), 'initial_lr': lr}], lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.EncStyle_opt = torch.optim.Adam([{'params': self.EncStyle.parameters(), 'initial_lr': lr}], lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.TransContent_opt = torch.optim.Adam([{'params': self.TransContent.parameters(), 'initial_lr': lr}], lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.Gen_opt = torch.optim.Adam([{'params': self.Gen.parameters(), 'initial_lr': lr}], lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

        # detector
        # self.Detector = networks.Detector()

        # Setup the loss function for training
        self.criterionL1 = torch.nn.L1Loss()

    def initialize(self):
        self.DisX.apply(networks.gaussian_weights_init)
        self.DisY.apply(networks.gaussian_weights_init)
        self.DisContent.apply(networks.gaussian_weights_init)
        # self.DisMaskContent.apply(networks.gaussian_weights_init)
        self.EncContent.apply(networks.gaussian_weights_init)
        self.DecMaskContent.apply(networks.gaussian_weights_init)
        self.EncStyle.apply(networks.gaussian_weights_init)
        self.Gen.apply(networks.gaussian_weights_init)

    def set_scheduler(self, opts, last_ep=0):
        self.DisX_sch = networks.get_scheduler(self.DisX_opt, opts, last_ep)
        self.DisY_sch = networks.get_scheduler(self.DisY_opt, opts, last_ep)
        self.DisContent_sch = networks.get_scheduler(self.DisContent_opt, opts, last_ep)
        self.DisMaskContentX_sch = networks.get_scheduler(self.DisMaskContentX_opt, opts, last_ep)
        self.DisMaskContentY_sch = networks.get_scheduler(self.DisMaskContentY_opt, opts, last_ep)
        self.EncContent_sch = networks.get_scheduler(self.EncContent_opt, opts, last_ep)
        self.DecMaskContent_sch = networks.get_scheduler(self.DecMaskContent_opt, opts, last_ep)
        self.EncStyle_sch = networks.get_scheduler(self.EncStyle_opt, opts, last_ep)
        self.Gen_sch = networks.get_scheduler(self.Gen_opt, opts, last_ep)

    def setgpu(self, gpu):
        self.gpu = gpu
        self.DisX.cuda(self.gpu)
        self.DisY.cuda(self.gpu)
        self.DisContent.cuda(self.gpu)
        self.DisMaskContentX.cuda(self.gpu)
        self.DisMaskContentY.cuda(self.gpu)
        self.EncContent.cuda(self.gpu)
        self.DecMaskContent.cuda(self.gpu)
        self.EncStyle.cuda(self.gpu)
        self.Gen.cuda(self.gpu)

    def update_lr(self):
        self.DisX_sch.step()
        self.DisY_sch.step()
        self.DisContent_sch.step()
        self.DisMaskContentX_sch.step()
        self.DisMaskContentY_sch.step()
        self.EncContent_sch.step()
        self.DecMaskContent_sch.step()
        self.EncStyle_sch.step()
        self.Gen_sch.step()

    def save(self, filename, ep, total_it):
        state = {
            'DisX': self.DisX.state_dict(),
            'DisY': self.DisY.state_dict(),
            'DisContent': self.DisContent.state_dict(),
            'DisMaskContentX': self.DisMaskContentX.state_dict(),
            'DisMaskContentY': self.DisMaskContentY.state_dict(),
            'EncContent': self.EncContent.state_dict(),
            'DecMaskContent': self.DecMaskContent.state_dict(),
            'EncStyle': self.EncStyle.state_dict(),
            'Gen': self.Gen.state_dict(),
            'DisX_opt': self.DisX_opt.state_dict(),
            'DisY_opt': self.DisY_opt.state_dict(),
            'DisContent_opt': self.DisContent_opt.state_dict(),
            'DisMaskContentX_opt': self.DisMaskContentX_opt.state_dict(),
            'DisMaskContentY_opt': self.DisMaskContentY_opt.state_dict(),
            'EncContent_opt': self.EncContent_opt.state_dict(),
            'DecMaskContent_opt': self.DecMaskContent_opt.state_dict(),
            'EncStyle_opt': self.EncStyle_opt.state_dict(),
            'Gen_opt': self.Gen_opt.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        torch.save(state, filename)
        return

    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir)
        # weight
        if train:
            self.DisContent.load_state_dict(checkpoint['DisContent'])
            self.DisX.load_state_dict(checkpoint['DisX'])
            self.DisY.load_state_dict(checkpoint['DisY'])
            self.DisMaskContentX.load_state_dict(checkpoint['DisMaskContentX'])
            self.DisMaskContentY.load_state_dict(checkpoint['DisMaskContentY'])
            self.DisContent_opt.load_state_dict(checkpoint['DisContent_opt'])
            self.DisX_opt.load_state_dict(checkpoint['DisX_opt'])
            self.DisY_opt.load_state_dict(checkpoint['DisY_opt'])
            self.DisMaskContentX_opt.load_state_dict(checkpoint['DisMaskContentX_opt'])
            self.DisMaskContentY_opt.load_state_dict(checkpoint['DisMaskContentY_opt'])
            self.EncContent_opt.load_state_dict(checkpoint['EncContent_opt'])
            self.DecMaskContent_opt.load_state_dict(checkpoint['DecMaskContent_opt'])
            self.EncStyle_opt.load_state_dict(checkpoint['EncStyle_opt'])
            self.Gen_opt.load_state_dict(checkpoint['Gen_opt'])
        self.EncContent.load_state_dict(checkpoint['EncContent'])
        self.DecMaskContent.load_state_dict(checkpoint['DecMaskContent'])
        self.EncStyle.load_state_dict(checkpoint['EncStyle'])
        self.Gen.load_state_dict(checkpoint['Gen'])

        return checkpoint['ep'], checkpoint['total_it']

    # TODO : 先不引入no ms
    # first do cloth transfer to get img a' with img a and mask, to do that we need to process mask to get cloth mask
    # second div a' into style and content and do cycle style transfer
    # I also need a detector to detect that person's position for further loss calculation and test process
    # TODO: add a detector and test the person's position and the positions in the generated results
    # TODO: add a decoder to visualize the process of converting mask?
    def forward(self):
        # get input images
        # for gpu capacity consideration, I only process one image at one time
        # if you want to process more images at a time, the code should do some changes first!
        half_size = 1
        # input x: N * C * H * W
        real_x = self.input_x
        real_y = self.input_y

        self.real_x = real_x[0:half_size]
        self.real_y = real_y[0:half_size]
        self.real_x_mask = self.mask_x[0:half_size]
        self.real_y_mask = self.mask_y[0:half_size]
        # self.real_x_box = self.box_x[0:half_size]
        # self.real_y_box = self.box_y[0:half_size]

        # TODO: too long processing way may be unuseful for the total process
        # content_x size: 1, 256, 64, 64
        self.content_x, self.content_y = self.EncContent(self.real_x, self.real_y)
        self.style_x, self.style_y = self.EncStyle(self.real_x, self.real_y)

        # random style code
        self.style_random = self.get_random(self.real_x.size(0), self.styleSpace, 'gauss')

        # do content transfer
        # convert a black-white img to 256 dimensions seems bad
        # x Encx-> y Decy-> y Ency->x Decx->x
        # y Ency-> x Decx-> x Encx->y Decy->y
        self.content_y_mask, self.content_x_mask = self.EncContent(self.real_x_mask, self.real_y_mask)
        # print('content_x_mask size:', self.content_x_mask.size())
        self.fake_x_mask, self.fake_y_mask = self.DecMaskContent(self.content_x_mask, self.content_y_mask)
        self.content_fake_y_mask, self.content_fake_x_mask = self.EncContent(self.fake_x_mask, self.fake_y_mask)
        self.recon_x_mask, self.recon_y_mask = self.DecMaskContent(self.content_fake_x_mask, self.content_fake_y_mask)
        self.self_content_y_mask, self.self_content_x_mask = self.EncContent(self.real_y_mask, self.real_x_mask)
        self.self_x_mask, self.self_y_mask = self.DecMaskContent(self.self_content_x_mask, self.self_content_y_mask)
        # print('fake_x_mask size:', self.fake_x_mask.size())
        # content_x_merge = torch.cat((self.content_x, self.content_x_mask), dim=1)
        # content_y_merge = torch.cat((self.content_y, self.content_y_mask), dim=1)
        # content_x_merge_output, content_y_merge_output = self.TransContent(content_x_merge, content_y_merge)
        # self.content_x_new, self.content_x_mask_new = torch.split(content_x_merge_output, self.content_x.size(1), dim=1)
        # self.content_y_new, self.content_y_mask_new = torch.split(content_y_merge_output, self.content_y.size(1), dim=1)
        # self.fake_x_mask_new, self.fake_y_mask_new = self.DecMaskContent(self.content_x_mask_new, self.content_y_mask_new)

        input_content_x = torch.cat((self.content_x, self.content_y, self.content_y), dim=0)
        input_style_x = torch.cat((self.style_x, self.style_x, self.style_random), dim=0)
        output_x = self.Gen.forward_x(input_content_x, input_style_x)
        self.self_recon_x, self.cross_fake_x, self.cross_fake_x_random = torch.split(output_x, self.content_x.size(0),
                                                                                     dim=0)

        input_content_y = torch.cat((self.content_y, self.content_x, self.content_x), dim=0)
        input_style_y = torch.cat((self.style_y, self.style_y, self.style_random), dim=0)
        output_y = self.Gen.forward_y(input_content_y, input_style_y)
        self.self_recon_y, self.cross_fake_y, self.cross_fake_y_random = torch.split(output_y, self.content_x.size(0),
                                                                                     dim=0)

        # this line looks odd but it means cross_fake_y is combined by content x and style y
        self.content_fake_x, self.content_fake_y = self.EncContent.forward(self.cross_fake_y, self.cross_fake_x)
        self.style_fake_x, self.style_fake_y = self.EncStyle.forward(self.cross_fake_x, self.cross_fake_y)

        # reconstruct images
        self.recon_fake_x = self.Gen.forward_x(self.content_fake_x, self.style_fake_x)
        self.recon_fake_y = self.Gen.forward_y(self.content_fake_y, self.style_fake_y)

        self.style_random_x, self.style_random_y = self.EncStyle(self.cross_fake_x_random, self.cross_fake_y_random)

        # for display TODO: need to show more generated results
        self.image_display = torch.cat((self.real_x[0:1].detach().cpu(), self.cross_fake_y[0:1].detach().cpu(),
                                        self.recon_fake_x[0:1].detach().cpu(),
                                        self.real_y[0:1].detach().cpu(), self.cross_fake_x[0:1].detach().cpu(),
                                        self.recon_fake_y[0:1].detach().cpu()), dim=0)

        self.mask_display = torch.cat((self.real_x_mask[0:1].detach().cpu(), self.fake_y_mask[0:1].detach().cpu(),
                                       self.recon_x_mask[0:1].detach().cpu(),
                                       self.real_y_mask[0:1].detach().cpu(), self.fake_x_mask[0:1].detach().cpu(),
                                       self.recon_y_mask[0:1].detach().cpu(),
                                       ), dim=0)

    def split(self, x):
        """Split data into image and mask(only for 3-channel image)"""
        return x[:, :3, :, :], x[:, 3:, :, :]

    def forward_Cloth(self):
        half_size = 1
        # input x: N * C * H * W
        real_x = self.input_x
        real_y = self.input_y

        self.real_x = real_x[0:half_size]
        self.real_y = real_y[0:half_size]
        self.real_x_mask = self.mask_x[0:half_size]
        self.real_y_mask = self.mask_y[0:half_size]
        self.real_x_box = self.box_x[0:half_size]
        self.real_y_box = self.box_y[0:half_size]

        # for ClothTransfer Net, we input the raw image and mask and get transfered image and mask back
        self.input_cloth_x = torch.cat((self.real_x, self.real_x_mask), dim=1)
        self.input_cloth_y = torch.cat((self.real_y, self.real_y_mask), dim=1)
        self.output_cloth_y = self.ClothTransfer.forward_x(self.input_cloth_x)
        self.output_recon_cloth_x = self.ClothTransfer.forward_y(self.output_cloth_y)
        self.output_cloth_x = self.ClothTransfer.forward_y(self.input_cloth_y)
        self.output_recon_cloth_y = self.ClothTransfer.forward_x(self.output_cloth_x)

        self.cloth_y, self.transfer_y_mask = self.split(self.output_cloth_y)
        self.cloth_x, self.transfer_x_mask = self.split(self.output_cloth_x)
        self.recon_cloth_x, self.recon_transfer_x_mask = self.split(self.output_recon_cloth_x)
        self.recon_cloth_y, self.recon_transfer_y_mask = self.split(self.output_recon_cloth_y)

        # print('input cloth x size:', self.input_cloth_x.size())
        # print('input cloth y size:', self.input_cloth_y.size())
        # print('output cloth x size:', self.output_cloth_x.size())
        # print('output cloth x size:', self.output_cloth_y.size())
        # print('output_recon_cloth_x size:', self.output_recon_cloth_x.size())
        # print('output_recon_cloth_y size:', self.output_recon_cloth_y.size())
        # print('cloth_y size:', self.cloth_y.size())
        # print('transfer_y_mask size:', self.transfer_y_mask.size())
        # print('cloth_x size:', self.cloth_x.size())
        # print('transfer_x_mask size:', self.transfer_x_mask.size())
        # print('recon_cloth_x size:', self.recon_cloth_x.size())
        # print('recon_transfer_x_mask size:', self.recon_transfer_x_mask.size())
        # print('recon_cloth_y size:', self.recon_cloth_y.size())
        # print('recon_transfer_y_mask size:', self.recon_transfer_y_mask.size()
        self.image_display = torch.cat((self.real_x[0:1].detach().cpu(), self.cloth_y[0:1].detach().cpu(),
                                        self.recon_cloth_x[0:1].detach().cpu(),
                                        self.real_y[0:1].detach().cpu(), self.cloth_x[0:1].detach().cpu(),
                                        self.recon_cloth_y[0:1].detach().cpu(),
                                        ), dim=0)

        self.image_mask_display = torch.cat((self.real_x_mask.detach().cpu(), self.transfer_y_mask.detach().cpu(),
                                             self.recon_transfer_x_mask.detach().cpu(),
                                             self.real_y_mask.detach().cpu(), self.transfer_x_mask.detach().cpu(),
                                             self.recon_transfer_y_mask.detach().cpu()
                                             ), dim=0)

    def fuseFeature(self, features, content, style, boxes, type='x'):
        new_contents = []
        for i in range(len(boxes)):
            if len(boxes[i]) > 0:
                feature = features[i:i+1]
                newmap = torch.zeros(feature.size())

                # exist instance(people) in this image
                # print('content size:', content[i][0].size())
                # print('style size:', style[i][0].size())
                new_ins = []
                for ins, s, box in zip(content[i], style[i], boxes[i]):
                    if type == 'x':
                        # new_content = self.GenIns.forward_x(ins)
                        # merge = self.InsResBlock.forward_x(new_content, s)
                        merge = self.InsResBlock.forward_x(ins, s)
                    elif type == 'y':
                        # new_content = self.GenIns.forward_y(ins)
                        # merge = self.InsResBlock.forward_y(new_content, s)
                        merge = self.InsResBlock.forward_y(ins, s)
                    else:
                        print('type error!')
                        break
                    new_ins.append(ins)

                    # merge into newmap with box
                    grid = torch.ones((newmap.size(0), newmap.size(2), newmap.size(3), 2))
                    grid = grid+1
                    box = box * newmap.size(2) / self.size
                    # int化会存在精度损失，直接覆盖feature可能会存在问题
                    box = (box+0.5).int()
                    xl, yl, xr, yr = box
                    dx = torch.linspace(-1, 1, xr-xl)
                    dy = torch.linspace(-1, 1, yr-yl)
                    dx = dx.unsqueeze(dim=0)
                    dy = dy.unsqueeze(dim=1)
                    # item()用于将tensor转换为int
                    dx = torch.repeat_interleave(dx, repeats=(yr - yl).item(), dim=0)
                    dy = torch.repeat_interleave(dy, repeats=(xr - xl).item(), dim=1)

                    grid[0, yl:yr, xl:xr, 0] = dx
                    grid[0, yl:yr, xl:xr, 1] = dy
                    # TODO align_corners的设定
                    output = F.grid_sample(merge, grid.to(device='cuda'), mode='bilinear', padding_mode='zeros', align_corners=False)
                    features[i][output[0]!=0] = output[0][output[0]!=0]
                new_contents.append(new_ins)
            else:
                new_contents.append([])

        if type == 'x':
            self.new_content_xi = new_contents
        elif type == 'y':
            self.new_content_yi = new_contents
        return features

    def set_requires_grad(self, net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad

    def update(self, images_x, images_y, masks_x, masks_y):
        self.input_x = images_x
        self.input_y = images_y
        self.mask_x = masks_x
        self.mask_y = masks_y
        # self.box_x = boxes_x
        # self.box_y = boxes_y
        # self.person_x = person_x
        # self.person_y = person_y

        self.forward()

        # update DisX
        self.DisX_opt.zero_grad()
        loss_D_X = self.backwardD(self.DisX, self.real_x, self.cross_fake_y)
        self.DisX_loss = loss_D_X.item()
        self.DisX_opt.step()

        # update DisY
        self.DisY_opt.zero_grad()
        loss_D_Y = self.backwardD(self.DisY, self.real_y, self.cross_fake_x)
        self.DisY_loss = loss_D_Y.item()
        self.DisY_opt.step()

        # update DisContent
        self.DisContent_opt.zero_grad()
        loss_DisContent = self.backwardDContent(self.DisContent, self.content_x, self.content_y)
        self.DisContent_loss = loss_DisContent.item()
        # nn.utils.clip_grad_norm_(self.DisContent.parameters(), 5)
        self.DisContent_opt.step()

        # update DisMaskContent
        self.DisMaskContentX_opt.zero_grad()
        loss_DisMaskContent_X = self.backwardD(self.DisMaskContentX, self.real_x_mask, self.fake_x_mask)
        self.DisMaskContent_loss_x = loss_DisMaskContent_X.item()
        self.DisMaskContentX_opt.step()

        self.DisMaskContentY_opt.zero_grad()
        loss_DisMaskContent_Y = self.backwardD(self.DisMaskContentY, self.real_y_mask, self.fake_y_mask)
        self.DisMaskContent_loss_y = loss_DisMaskContent_Y.item()
        self.DisMaskContentY_opt.step()

        # update Encoder and Gen
        self.EncContent_opt.zero_grad()
        # self.EncMaskContent_opt.zero_grad()
        self.DecMaskContent_opt.zero_grad()
        self.EncStyle_opt.zero_grad()
        # self.TransContent_opt.zero_grad()
        self.Gen_opt.zero_grad()
        self.backwardEG()
        self.EncContent_opt.step()
        # self.EncMaskContent_opt.step()
        self.DecMaskContent_opt.step()
        self.EncStyle_opt.step()
        # self.TransContent_opt.step()
        self.Gen_opt.step()

    def update_Cloth(self, images_x, images_y, masks_x, masks_y, boxes_x, boxes_y):
        self.input_x = images_x
        self.input_y = images_y
        self.mask_x = masks_x
        self.mask_y = masks_y
        self.box_x = boxes_x
        self.box_y = boxes_y

        self.forward_Cloth()

        # update ClothTransfer DONE
        self.set_requires_grad(self.DisClothTransfer, False)
        self.ClothTransfer_opt.zero_grad()
        self.backwardClothTransfer()
        self.ClothTransfer_opt.step()

        # update DisClothTransfer DONE
        self.set_requires_grad(self.DisClothTransfer, True)
        self.DisClothTransfer_opt.zero_grad()
        self.backwardDisClothTransfer()
        self.DisClothTransfer_opt.step()

    def backwardClothTransfer(self):
        lambda_X = 10.0
        lambda_Y = 10.0
        lambda_idt = 1.0
        lambda_ctx = 1.0

        # print('device:', self.output_cloth_x.device)
        loss_CT_x = self.criterionGAN(self.DisClothTransfer.forward_x(self.output_cloth_y.detach()), True)
        loss_CT_y = self.criterionGAN(self.DisClothTransfer.forward_y(self.output_cloth_x.detach()), True)
        loss_CT_cyc_x = self.criterionCyc(self.output_recon_cloth_x, self.input_cloth_x) * lambda_X
        loss_CT_cyc_y = self.criterionCyc(self.output_recon_cloth_y, self.input_cloth_y) * lambda_Y
        loss_CT_idt_x = self.criterionIdt(self.ClothTransfer.forward_x(self.input_cloth_y),
                                          self.input_cloth_y.detach()) * lambda_X * lambda_idt
        loss_CT_idt_y = self.criterionIdt(self.ClothTransfer.forward_y(self.input_cloth_x),
                                          self.input_cloth_x.detach()) * lambda_Y * lambda_idt
        weight_x = self.get_weight_for_ctx(self.real_x_mask, self.transfer_y_mask)
        loss_CT_ctx_x = self.weighted_L1_loss(self.real_x, self.cloth_y, weight=weight_x) * lambda_X * lambda_ctx
        weight_y = self.get_weight_for_ctx(self.real_y_mask, self.transfer_x_mask)
        loss_CT_ctx_y = self.weighted_L1_loss(self.real_y, self.cloth_x, weight=weight_y) * lambda_Y * lambda_ctx
        self.CT_loss = loss_CT_x + loss_CT_y + loss_CT_cyc_x + loss_CT_cyc_y +\
                       loss_CT_idt_x + loss_CT_idt_y + loss_CT_ctx_x + loss_CT_ctx_y
        self.CT_loss.backward()

    def backwardDisClothTransfer(self):
        pred_real_x = self.DisClothTransfer.forward_x(self.input_cloth_x.detach())
        loss_Dx_real = self.criterionGAN(pred_real_x, True)
        pred_real_y = self.DisClothTransfer.forward_y(self.input_cloth_y.detach())
        loss_Dy_real = self.criterionGAN(pred_real_y, True)
        pred_fake_x = self.DisClothTransfer.forward_x(self.output_cloth_x.detach())
        loss_Dx_fake = self.criterionGAN(pred_fake_x, False)
        pred_fake_y = self.DisClothTransfer.forward_y(self.output_cloth_y.detach())
        loss_Dy_fake = self.criterionGAN(pred_fake_y, False)
        self.DisCT_x_loss = (loss_Dx_real + loss_Dx_fake) * 0.5
        self.DisCT_x_loss.backward()
        self.DisCT_y_loss = (loss_Dy_real + loss_Dy_fake) * 0.5
        self.DisCT_y_loss.backward()

    def get_weight_for_ctx(self, x, y):
        """Get weight for context preserving loss"""
        z = self.merge_masks(torch.cat([x, y], dim=1))
        return (1 - z) / 2  # [-1,1] -> [1,0]

    def merge_masks(self, segs):
        """Merge masks (B, N, W, H) -> (B, 1, W, H)"""
        ret = torch.sum((segs+1)/2, dim=1, keepdim=True)  # (B, 1, W, H)
        return ret.clamp(max=1, min=0) * 2 - 1

    def weighted_L1_loss(self, src, tgt, weight):
        """L1 loss with given weight (used for context preserving loss)"""
        return torch.mean(weight * torch.abs(src - tgt))

    def backwardD(self, netD, real, fake):
        pred_fake = netD.forward(fake.detach())
        pred_real = netD.forward(real)
        loss = 0
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = torch.sigmoid(out_a)
            out_real = torch.sigmoid(out_b)
            all0 = torch.zeros_like(out_fake).cuda(self.gpu)
            all1 = torch.ones_like(out_real).cuda(self.gpu)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            loss = loss + ad_fake_loss + ad_true_loss
        loss.backward()
        return loss

    def backwardDContent(self, netD, x, y):
        pred_x = netD.forward(x.detach())
        pred_y = netD.forward(y.detach())
        loss = 0
        for it, (x, y) in enumerate(zip(pred_x, pred_y)):
            out_x = torch.sigmoid(x)
            out_y = torch.sigmoid(y)
            all1 = torch.ones((out_y.size(0))).cuda(self.gpu)
            all0 = torch.zeros((out_x.size(0))).cuda(self.gpu)
            ad_y_loss = nn.functional.binary_cross_entropy(out_y, all1)
            ad_x_loss = nn.functional.binary_cross_entropy(out_x, all0)
            loss = loss + ad_y_loss + ad_x_loss
        loss.backward()
        return loss

    def backwardDInsContent(self, netD, xs, ys):
        loss = 0
        for x, y in zip(xs, ys):
            minLen = min(len(x), len(y))
            for i in range(minLen):
                insx = x[i]
                insy = y[i]
                pred_x = netD.forward(insx.detach())
                pred_y = netD.forward(insy.detach())
                for it, (px, py) in enumerate(zip(pred_x, pred_y)):
                    out_x = torch.sigmoid(px)
                    out_y = torch.sigmoid(py)
                    all1 = torch.ones((out_y.size(0))).cuda(self.gpu)
                    all0 = torch.zeros((out_x.size(0))).cuda(self.gpu)
                    ad_y_loss = nn.functional.binary_cross_entropy(out_y, all1)
                    ad_x_loss = nn.functional.binary_cross_entropy(out_x, all0)
                    loss = loss + ad_y_loss + ad_x_loss
        return loss

    def backwardEG(self):
        # content adv loss for gen
        loss_Gen_content_x = self.backwardG_content(self.content_x)
        loss_Gen_content_y = self.backwardG_content(self.content_y)

        # domain adversarial loss
        loss_Gen_X = self.backwardG(self.cross_fake_y, self.DisX)
        loss_Gen_Y = self.backwardG(self.cross_fake_x, self.DisY)

        # KL loss
        loss_KL_style_x = self._l2_regularize(self.style_x) * 0.01
        loss_KL_style_y = self._l2_regularize(self.style_y) * 0.01
        loss_KL_content_x = self._l2_regularize(self.content_x) * 0.01
        loss_KL_content_y = self._l2_regularize(self.content_y) * 0.01

        # cross cycle consistency loss
        loss_Gen_L1_recon_x = self.criterionL1(self.recon_fake_x, self.real_x) * 10
        loss_Gen_L1_recon_y = self.criterionL1(self.recon_fake_y, self.real_y) * 10

        # self-reconstruction loss
        loss_Gen_L1_self_x = self.criterionL1(self.self_recon_x, self.real_x) * 10
        loss_Gen_L1_self_y = self.criterionL1(self.self_recon_y, self.real_y) * 10

        # latent regression loss
        loss_LR_x = torch.mean(torch.abs(self.style_random_x - self.style_random)) * 10
        loss_LR_y = torch.mean(torch.abs(self.style_random_y - self.style_random)) * 10

        # content convert loss
        # loss_TransCon_x = self.criterionL1(self.content_x_mask_new, self.content_y_mask) * 30
        # loss_TransCon_y = self.criterionL1(self.content_y_mask_new, self.content_x_mask) * 30

        loss_Mask_x = self.criterionL1(self.fake_y_mask, self.real_y_mask) * 10
        loss_Mask_y = self.criterionL1(self.fake_x_mask, self.real_x_mask) * 10
        loss_recon_Mask_x = self.criterionL1(self.recon_x_mask, self.real_x_mask) * 10
        loss_recon_Mask_y = self.criterionL1(self.recon_y_mask, self.real_y_mask) * 10
        loss_self_Mask_x = self.criterionL1(self.self_x_mask, self.real_x_mask) * 10
        loss_self_Mask_y = self.criterionL1(self.self_y_mask, self.real_y_mask) * 10

        loss_DM_x = self.backwardG(self.fake_x_mask, self.DisMaskContentX)
        loss_DM_y = self.backwardG(self.fake_y_mask, self.DisMaskContentY)

        # new mask loss
        # loss_NMask_x = self.criterionL1(self.fake_x_mask_new, self.real_y_mask) * 50
        # loss_NMask_y = self.criterionL1(self.fake_y_mask_new, self.real_x_mask) * 50
        # loss_Mask_x_1 = torch.abs((torch.sum(self.fake_x_mask > 0) - torch.sum(self.real_y_mask > 0))) * 0.003
        # loss_Mask_y_1 = torch.abs((torch.sum(self.fake_y_mask > 0) - torch.sum(self.real_x_mask > 0))) * 0.003

        # content_consistency loss
        # loss_CC_content_x = self.criterionL1(self.content_x_new, self.content_x) * 10
        # loss_CC_content_y = self.criterionL1(self.content_y_new, self.content_y) * 10


        loss_Gen = loss_Gen_content_x + loss_Gen_content_y + loss_Gen_X + loss_Gen_Y + \
                   loss_KL_content_x + loss_KL_content_y + loss_KL_style_x + loss_KL_style_y + \
                   loss_Gen_L1_recon_x + loss_Gen_L1_recon_y + loss_Gen_L1_self_x + loss_Gen_L1_self_y + \
                   loss_LR_x + loss_LR_y + loss_Mask_x + loss_Mask_y +\
                   loss_recon_Mask_x + loss_recon_Mask_y + loss_self_Mask_x + loss_self_Mask_y + loss_DM_x + loss_DM_y

        loss_Gen.backward()

        self.Gen_loss_X = loss_Gen_X.item()
        self.Gen_loss_Y = loss_Gen_Y.item()
        self.Gen_loss_content_x = loss_Gen_content_x.item()
        self.Gen_loss_content_y = loss_Gen_content_y.item()
        self.KL_loss_style_x = loss_KL_style_x.item()
        self.KL_loss_style_y = loss_KL_style_y.item()
        self.KL_loss_content_x = loss_KL_content_x.item()
        self.KL_loss_content_y = loss_KL_content_y.item()
        self.L1_loss_recon_x = loss_Gen_L1_recon_x.item()
        self.L1_loss_recon_y = loss_Gen_L1_recon_y.item()
        self.L1_loss_self_x = loss_Gen_L1_self_x.item()
        self.L1_loss_self_y = loss_Gen_L1_self_y.item()
        self.Gen_loss = loss_Gen.item()
        self.LR_loss_x = loss_LR_x.item()
        self.LR_loss_y = loss_LR_y.item()
        # self.TransCon_loss_x = loss_TransCon_x.item()
        # self.TransCon_loss_y = loss_TransCon_y.item()
        self.Mask_loss_x = loss_Mask_x.item()
        self.Mask_loss_y = loss_Mask_y.item()
        # self.Mask_loss_x_1 = loss_Mask_x_1.item()
        # self.Mask_loss_y_1 = loss_Mask_y_1.item()
        self.Mask_recon_loss_x = loss_recon_Mask_x.item()
        self.Mask_recon_loss_y = loss_recon_Mask_y.item()
        self.Mask_self_loss_x = loss_self_Mask_x.item()
        self.Mask_self_loss_y = loss_self_Mask_y.item()
        self.DM_loss_x = loss_DM_x.item()
        self.DM_loss_y = loss_DM_y.item()
        # self.NMask_loss_x = loss_NMask_x.item()
        # self.NMask_loss_y = loss_NMask_y.item()
        # self.CC_loss_x = loss_CC_content_x.item()
        # self.CC_loss_y = loss_CC_content_y.item()

    def backwardG(self, fake_image, netD):
        outs_fake = netD.forward(fake_image)
        loss = 0
        for out in outs_fake:
            fake = torch.sigmoid(out)
            all_ones = torch.ones_like(fake).cuda(self.gpu)
            loss = loss + nn.functional.binary_cross_entropy(fake, all_ones)
        return loss

    def backwardG_content(self, content):
        outs = self.DisContent.forward(content)
        loss = 0
        for out in outs:
            outputs_fake = torch.sigmoid(out)
            all_half = 0.5 * torch.ones((outputs_fake.size(0))).cuda(self.gpu)
            loss = loss + nn.functional.binary_cross_entropy(outputs_fake, all_half)
        return loss

    def backwardInsGx(self, contents, netD):
        loss = 0
        for content in contents:
            for ins in content:
                pred_x = netD.forward(ins)
                for out in pred_x:
                    out_fake = torch.sigmoid(out)
                    all_zeros = torch.zeros((out_fake.size(0))).cuda(self.gpu)
                    loss = loss + nn.functional.binary_cross_entropy(out_fake, all_zeros)
        return loss

    def backwardInsGy(self, contents, netD):
        loss = 0
        for content in contents:
            for ins in content:
                pred_y = netD.forward(ins)
                for out in pred_y:
                    out_fake = torch.sigmoid(out)
                    all_ones = torch.ones((out_fake.size(0))).cuda(self.gpu)
                    loss = loss + nn.functional.binary_cross_entropy(out_fake, all_ones)
        return loss

    def _l2_regularize(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def get_random(self, batchsize, nz, random_type='gauss'):
        style = torch.randn(batchsize, nz).cuda(self.gpu)
        return style

    def assemble_outputs(self):
        # TODO: do whole process and output
        images_x = self.normalize_image(self.real_x).detach()
        images_y = self.normalize_image(self.real_y).detach()
        images_u = self.normalize_image(self.cross_fake_y).detach()
        images_v = self.normalize_image(self.cross_fake_x).detach()
        images_recon_x = self.normalize_image(self.recon_fake_x).detach()
        images_recon_y = self.normalize_image(self.recon_fake_y).detach()
        row1 = torch.cat((images_x[0:1, ::], images_u[0:1, ::], images_recon_x[0:1, ::]), dim=3)
        row2 = torch.cat((images_y[0:1, ::], images_v[0:1, ::], images_recon_y[0:1, ::]), dim=3)
        return torch.cat((row1, row2), dim=2)

    def normalize_image(self, x):
        return x[:, 0:3, :, :]





