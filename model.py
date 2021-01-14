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
        channel = 256

        # discriminators 包括区分图片是否是生成的Dx, Dy
        # 区分content属于哪一类的Dc 和 Dic(这两者损失函数不同)
        self.DisX = networks.Discriminator(opts.input_dim_x, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
        self.DisY = networks.Discriminator(opts.input_dim_y, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
        self.DisContent = networks.ContentDiscriminator()
        self.DisInsContent = networks.ContentInsDiscriminator()

        # encoders 包括style encoders, content encoders
        # 目前采用的是INIT中的实现方式，即Exc和Exci共用一个encoder
        self.EncContent = networks.ContentEncoder(opts.input_dim_x, opts.input_dim_y)
        self.EncStyle = networks.StyleEncoder(opts.input_dim_x, opts.input_dim_y, self.styleSpace)

        # residual block 包括global-level和instance-level, 用来融合content feature和style feature
        self.GloResBlock = networks.GlobalLevelResBlock(channel, channel)
        self.InsResBlock = networks.InstanceLevelResBlock(channel, channel)

        # generator
        self.Gen = networks.Generator(opts.input_dim_x, opts.input_dim_y, nz=self.styleSpace)
        input_nc = 256
        output_nc = 256
        ngf = 64
        norm_layer = 'instance'
        use_dropout = False
        self.GenIns = networks.ResnetGenerator(input_nc, output_nc, ngf)

        # optimizers
        self.DisX_opt = torch.optim.Adam(self.DisX.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.DisY_opt = torch.optim.Adam(self.DisY.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.DisContent_opt = torch.optim.Adam(self.DisContent.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.DisInsContent_opt = torch.optim.Adam(self.DisInsContent.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.EncContent_opt = torch.optim.Adam(self.EncContent.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.EncStyle_opt = torch.optim.Adam(self.EncStyle.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.GloResBlock_opt = torch.optim.Adam(self.GloResBlock.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.InsResBlock_opt = torch.optim.Adam(self.GloResBlock.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.Gen_opt = torch.optim.Adam(self.Gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.GenIns_opt = torch.optim.Adam(self.GenIns.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

        # detector
        self.Detector = networks.Detector()

        # Setup the loss function for training
        self.criterionL1 = torch.nn.L1Loss()

    def initialize(self):
        self.DisX.apply(networks.gaussian_weights_init)
        self.DisY.apply(networks.gaussian_weights_init)
        self.DisContent.apply(networks.gaussian_weights_init)
        self.DisInsContent.apply(networks.gaussian_weights_init)
        self.EncContent.apply(networks.gaussian_weights_init)
        self.EncStyle.apply(networks.gaussian_weights_init)
        self.GloResBlock.apply(networks.gaussian_weights_init)
        self.InsResBlock.apply(networks.gaussian_weights_init)
        self.Gen.apply(networks.gaussian_weights_init)
        self.GenIns.apply(networks.gaussian_weights_init)
        self.Detector.eval()

    def set_scheduler(self, opts, last_ep=0):
        self.DisX_sch = networks.get_scheduler(self.DisX_opt, opts, last_ep)
        self.DisY_sch = networks.get_scheduler(self.DisY_opt, opts, last_ep)
        self.DisContent_sch = networks.get_scheduler(self.DisContent_opt, opts, last_ep)
        self.DisInsContent_sch = networks.get_scheduler(self.DisInsContent_opt, opts, last_ep)
        self.EncContent_sch = networks.get_scheduler(self.EncContent_opt, opts, last_ep)
        self.EncStyle_sch = networks.get_scheduler(self.EncStyle_opt, opts, last_ep)
        self.GloResBlock_sch = networks.get_scheduler(self.GloResBlock_opt, opts, last_ep)
        self.InsResBlock_sch = networks.get_scheduler(self.InsResBlock_opt, opts, last_ep)
        self.Gen_sch = networks.get_scheduler(self.Gen_opt, opts, last_ep)
        self.GenIns_sch = networks.get_scheduler(self.GenIns_opt, opts, last_ep)

    def setgpu(self, gpu):
        self.gpu = gpu
        self.DisX.cuda(self.gpu)
        self.DisY.cuda(self.gpu)
        self.DisContent.cuda(self.gpu)
        self.DisInsContent.cuda(self.gpu)
        self.EncContent.cuda(self.gpu)
        self.EncStyle.cuda(self.gpu)
        self.GloResBlock.cuda(self.gpu)
        self.InsResBlock.cuda(self.gpu)
        self.Gen.cuda(self.gpu)
        self.GenIns.cuda(self.gpu)
        self.Detector.cuda(self.gpu)

    def update_lr(self):
        self.DisX_sch.step()
        self.DisY_sch.step()
        self.DisContent_sch.step()
        self.DisInsContent_sch.step()
        self.EncContent_sch.step()
        self.EncStyle_sch.step()
        self.Gen_sch.step()
        self.GenIns_sch.step()

    # TODO : 先不引入no ms
    def forward(self):
        # get input images
        half_size = 1
        # input x: N * C * H * W
        real_x = self.input_x
        real_y = self.input_y

        self.real_x = real_x[0:half_size]
        self.real_y = real_y[0:half_size]

        self.content_x, self.content_y = self.EncContent(self.real_x, self.real_y)
        self.style_x, self.style_y = self.EncStyle(self.real_x, self.real_y)

        # random style code
        self.style_random = self.get_random(self.real_x.size(0), self.styleSpace, 'gauss')

        merge_x_random = self.GloResBlock.forward_x(self.content_y, self.style_random)
        merge_y_random = self.GloResBlock.forward_x(self.content_x, self.style_random)

        # self reconstruct
        merge_x = self.GloResBlock.forward_x(self.content_x, self.style_x)
        merge_y = self.GloResBlock.forward_y(self.content_y, self.style_y)
        # self.fake_x = self.Gen.forward_x(merge_x)
        # self.fake_y = self.Gen.forward_y(merge_y)

        # cross
        merge_u = self.GloResBlock.forward_x(self.content_x, self.style_y)
        merge_v = self.GloResBlock.forward_y(self.content_y, self.style_x)

        # get instance(people) in input
        with torch.no_grad():
            real_x_instances, x_boxes = self.Detector(self.real_x)
            real_y_instances, y_boxes = self.Detector(self.real_y)
            # content_xi 由N张图组成，每张图中由m个instance组成，每个instance是
            # 一个四维tensor N(1) * C(256) * H * W
            self.content_xi = [
                [self.EncContent.forward_x(x.to(device='cuda')) for x in x_ins]
                for x_ins in real_x_instances
            ]
            self.content_yi = [
                [self.EncContent.forward_y(y.to(device='cuda')) for y in y_ins]
                for y_ins in real_y_instances
            ]
            self.style_xi = [
                [self.EncStyle.forward_x(x.to(device='cuda')) for x in x_ins]
                for x_ins in real_x_instances
            ]
            self.style_yi = [
                [self.EncStyle.forward_y(y.to(device='cuda')) for y in y_ins]
                for y_ins in real_y_instances
            ]

        merge_u = self.fuseFeature(merge_u, self.content_xi, self.style_xi, x_boxes, type='x')
        merge_v = self.fuseFeature(merge_v, self.content_yi, self.style_yi, y_boxes, type='y')

        input_x = torch.cat((merge_v, merge_x, merge_x_random), dim=0)
        input_y = torch.cat((merge_u, merge_y, merge_y_random), dim=0)
        output_x = self.Gen.forward_x(input_x)
        output_y = self.Gen.forward_y(input_y)

        # fake_u 由content x和style y 组成， fake_v由content y和style x 组成
        self.fake_v, self.fake_x, self.fake_x_random = torch.split(output_x, merge_x.size(0), dim=0)
        self.fake_u, self.fake_y, self.fake_y_random = torch.split(output_y, merge_y.size(0), dim=0)

        # self.fake_u = self.Gen.forward_y(merge_u)
        # self.fake_v = self.Gen.forward_x(merge_v)

        # content_u <-> content_x, content_v <-> content_y, style_u <-> style_y, style_v <-> style_x
        self.content_u, self.content_v = self.EncContent(self.fake_u, self.fake_v)
        self.style_v, self.style_u = self.EncStyle(self.fake_v, self.fake_u)

        # reconstruct
        merge_x_re = self.GloResBlock.forward_x(self.content_u, self.style_v)
        merge_y_re = self.GloResBlock.forward_y(self.content_v, self.style_u)
        self.recon_x = self.Gen.forward_x(merge_x_re)
        self.recon_y = self.Gen.forward_y(merge_y_re)

        # latent regression
        self.style_random_x, self.style_random_y = self.EncStyle.forward(self.fake_x_random, self.fake_y_random)

        # for display
        self.image_display = torch.cat((self.real_x[0:1].detach().cpu(), self.fake_u[0:1].detach().cpu(), \
                                        self.fake_y_random[0:1].detach().cpu(), self.fake_x[0:1].detach().cpu(), self.recon_x[0:1].detach().cpu(), \
                                        self.real_y[0:1].detach().cpu(), self.fake_v[0:1].detach().cpu(), \
                                        self.fake_x_random[0:1].detach().cpu(), self.fake_y[0:1].detach().cpu(), self.recon_y[0:1].detach().cpu()), dim=0)

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


    def update(self, images_x, images_y):
        self.input_x = images_x
        self.input_y = images_y

        self.forward()

        # update DisX
        self.DisX_opt.zero_grad()
        loss_D_X = self.backwardD(self.DisX, self.real_x, self.fake_v)
        self.DisX_loss = loss_D_X.item()
        self.DisX_opt.step()

        # update DisY
        self.DisY_opt.zero_grad()
        loss_D_Y = self.backwardD(self.DisY, self.real_y, self.fake_u)
        self.DisY_loss = loss_D_Y.item()
        self.DisY_opt.step()

        # update DisContent
        self.DisContent_opt.zero_grad()
        loss_DisContent = self.backwardDContent(self.DisContent, self.content_x, self.content_y)
        self.DisContent_loss = loss_DisContent.item()
        # nn.utils.clip_grad_norm_(self.DisContent.parameters(), 5)
        self.DisContent_opt.step()

        # update DisInsContent
        # self.DisInsContent_opt.zero_grad()
        # loss_DisInsContent_x = self.backwardDInsContent(self.DisInsContent, self.content_xi, self.new_content_xi)
        # loss_DisInsContent_y = self.backwardDInsContent(self.DisInsContent, self.new_content_yi, self.content_yi)
        # loss_DisInsContent = loss_DisInsContent_x + loss_DisInsContent_y
        # if loss_DisInsContent != 0:
        #     loss_DisInsContent.backward()
        #     self.DisInsContent_loss = loss_DisInsContent.item()
        # else:
        #     self.DisInsContent_loss = 0
        # self.DisInsContent_opt.step()

        # update Encoder and Gen
        self.EncContent_opt.zero_grad()
        self.EncStyle_opt.zero_grad()
        self.GloResBlock_opt.zero_grad()
        self.Gen_opt.zero_grad()
        self.backwardEG()
        self.EncContent_opt.step()
        self.EncStyle_opt.step()
        self.GloResBlock_opt.step()
        self.Gen_opt.step()

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
        loss_Gen_X = self.backwardG(self.fake_v, self.DisX)
        loss_Gen_Y = self.backwardG(self.fake_u, self.DisY)

        # KL loss
        loss_KL_style_x = self._l2_regularize(self.style_x) * 0.01
        loss_KL_style_y = self._l2_regularize(self.style_y) * 0.01
        loss_KL_content_x = self._l2_regularize(self.content_x) * 0.01
        loss_KL_content_y = self._l2_regularize(self.content_y) * 0.01

        # cross cycle consistency loss
        loss_Gen_L1_recon_x = self.criterionL1(self.recon_x, self.real_x) * 10
        loss_Gen_L1_recon_y = self.criterionL1(self.recon_y, self.real_y) * 10

        # self-reconstruction loss
        loss_Gen_L1_self_x = self.criterionL1(self.fake_x, self.real_x) * 10
        loss_Gen_L1_self_y = self.criterionL1(self.fake_y, self.real_y) * 10

        # latent regression loss
        loss_LR_x = torch.mean(torch.abs(self.style_random_x - self.style_random)) * 10
        loss_LR_y = torch.mean(torch.abs(self.style_random_y - self.style_random)) * 10

        # instance feature generator loss
        # loss_InsGen_Y = self.backwardInsGy(self.new_content_xi, self.DisInsContent) * 50
        # loss_InsGen_X = self.backwardInsGx(self.new_content_yi, self.DisInsContent) * 50

        # loss_Gen = loss_Gen_content_x + loss_Gen_content_y + loss_Gen_X + loss_Gen_Y + \
        #     loss_KL_content_x + loss_KL_content_y + loss_KL_style_x + loss_KL_style_y + \
        #     loss_Gen_L1_recon_x + loss_Gen_L1_recon_y + loss_Gen_L1_self_x + loss_Gen_L1_self_y + \
        #     loss_LR_x + loss_LR_y + loss_InsGen_X + loss_InsGen_Y
        loss_Gen = loss_Gen_content_x + loss_Gen_content_y + loss_Gen_X + loss_Gen_Y + \
                   loss_KL_content_x + loss_KL_content_y + loss_KL_style_x + loss_KL_style_y + \
                   loss_Gen_L1_recon_x + loss_Gen_L1_recon_y + loss_Gen_L1_self_x + loss_Gen_L1_self_y + \
                   loss_LR_x + loss_LR_y
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

    def save(self, filename, ep, total_it):
        state = {
            'DisX': self.DisX.state_dict(),
            'DisY': self.DisY.state_dict(),
            'DisContent': self.DisContent.state_dict(),
            'DisInsContent': self.DisInsContent.state_dict(),
            'EncContent': self.EncContent.state_dict(),
            'EncStyle': self.EncStyle.state_dict(),
            'Gen': self.Gen.state_dict(),
            'GlobalResBlock': self.GloResBlock.state_dict(),
            'InsResBlock': self.InsResBlock.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        torch.save(state, filename)
        return

    def assemble_outputs(self):
        images_x = self.normalize_image(self.real_x).detach()
        images_y = self.normalize_image(self.real_y).detach()
        images_u = self.normalize_image(self.fake_u).detach()
        images_v = self.normalize_image(self.fake_v).detach()
        images_fakex = self.normalize_image(self.fake_x).detach()
        images_fakey = self.normalize_image(self.fake_y).detach()
        images_reconx = self.normalize_image(self.recon_x).detach()
        images_recony = self.normalize_image(self.recon_y).detach()
        row1 = torch.cat((images_x[0:1, ::], images_u[0:1, ::], images_reconx[0:1, ::], images_fakex[0:1, ::]), dim=3)
        row2 = torch.cat((images_y[0:1, ::], images_v[0:1, ::], images_recony[0:1, ::], images_fakey[0:1, ::]), dim=3)
        return torch.cat((row1, row2), dim=2)

    def normalize_image(self, x):
        return x[:, 0:3, :, :]





