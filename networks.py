import torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F

from torchvision.models import resnet50
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor

####################################################################
#------------------------- Discriminators --------------------------
####################################################################
class Discriminator(nn.Module):
    # 'sn' means spectral normalization
    def __init__(self, input_dim, norm='None', sn=False):
        super(Discriminator, self).__init__()
        channel = 64
        n_layer = 6
        self.model = self._make_net(channel, input_dim, n_layer, norm, sn)

    def _make_net(self, ch, input_dim, n_layer, norm, sn):
        model = []
        # n layers LeakyReLUConv2d
        model += [LeakyReLUConv2d(input_dim, ch, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)]
        for i in range(1, n_layer-1):
            model += [LeakyReLUConv2d(ch, ch*2, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)]
            ch = ch * 2
        model += [LeakyReLUConv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1, norm='None', sn=sn)]
        ch = ch * 2
        if sn:
            model += [nn.utils.spectral_norm(nn.Conv2d(ch, 1, kernel_size=1, stride=1, padding=0))]
        else:
            model += [nn.Conv2d(ch, 1, kernel_size=1, stride=1, padding=0)]
        return nn.Sequential(*model)

    def cuda(self, gpu):
        self.model.cuda(gpu)

    def forward(self, x):
        out = self.model(x)
        out = out.view(-1)
        outs = []
        outs.append(out)
        return outs

class ContentDiscriminator(nn.Module):
    def __init__(self):
        super(ContentDiscriminator, self).__init__()
        self.model = nn.Sequential(
            LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1, norm='Instance'),
            LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1, norm='Instance'),
            LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1, norm='Instance'),
            LeakyReLUConv2d(256, 256, kernel_size=4, stride=1, padding=0),
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        out = self.model(x)
        out = out.view(-1)
        outs = []
        outs.append(out)
        return outs

class ContentInsDiscriminator(nn.Module):
    def __init__(self):
        super(ContentInsDiscriminator, self).__init__()
        self.model = nn.Sequential(
            LeakyReLUConv2d(256, 256, kernel_size=3, stride=2, padding=1, norm='Instance'),
            LeakyReLUConv2d(256, 256, kernel_size=3, stride=2, padding=1, norm='Instance'),
            LeakyReLUConv2d(256, 256, kernel_size=3, stride=2, padding=1, norm='Instance'),
            LeakyReLUConv2d(256, 256, kernel_size=2, stride=1, padding=0),
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        out = self.model(x)
        out = out.view(-1)
        outs = []
        outs.append(out)
        return outs

####################################################################
#---------------------------- Encoders -----------------------------
####################################################################
class ContentEncoder(nn.Module):
    def __init__(self, input_dim_x, input_dim_y):
        super(ContentEncoder, self).__init__()
        enc_X = []
        enc_Y = []
        dim = 64
        enc_X += [LeakyReLUConv2d(input_dim_x, dim, kernel_size=7, stride=1, padding=3)]
        enc_Y += [LeakyReLUConv2d(input_dim_y, dim, kernel_size=7, stride=1, padding=3)]
        for i in range(1, 3):
            enc_X += [ReLUINSConv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)]
            enc_Y += [ReLUINSConv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)]
            dim *= 2
        for i in range(0, 3):
            enc_X += [INSResBlock(dim, dim)]
            enc_Y += [INSResBlock(dim, dim)]

        self.convX = nn.Sequential(*enc_X)
        self.convY = nn.Sequential(*enc_Y)

        enc_share = []
        for i in range(0, 1):
            enc_share += [INSResBlock(dim, dim)]
            enc_share += [GaussianNoiseLayer()]
        self.conv_share = nn.Sequential(*enc_share)

    def forward(self, x, y):
        outX = self.convX(x)
        outY = self.convY(y)
        outX = self.conv_share(outX)
        outY = self.conv_share(outY)
        return outX, outY

    def forward_x(self, x):
        out = self.convX(x)
        out = self.conv_share(out)
        return out

    def forward_y(self, y):
        out = self.convY(y)
        out = self.conv_share(out)
        return out

class StyleEncoder(nn.Module):
    def __init__(self, input_dim_x, input_dim_y, output=8):
        super(StyleEncoder, self).__init__()
        dim = 64
        self.model_x = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_dim_x, dim, 7, 1),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim * 2, 4, 2),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim * 2, dim * 4, 4, 2),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim * 4, dim * 4, 4, 2),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim * 4, dim * 4, 4, 2),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 4, output, 1, 1, 0)
        )
        self.model_y = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_dim_y, dim, 7, 1),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim * 2, 4, 2),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim * 2, dim * 4, 4, 2),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim * 4, dim * 4, 4, 2),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim * 4, dim * 4, 4, 2),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 4, output, 1, 1, 0)
        )
        return

    def forward(self, x, y):
        out_x = self.model_x(x)
        out_y = self.model_y(y)
        out_x = out_x.view(out_x.size(0), -1)
        out_y = out_y.view(out_y.size(0), -1)
        return out_x, out_y

    def forward_x(self, x):
        out_x = self.model_x(x)
        out_x = out_x.view(out_x.size(0), -1)
        return out_x

    def forward_y(self, y):
        out_y = self.model_y(y)
        out_y = out_y.view(out_y.size(0), -1)
        return out_y

####################################################################
#--------------------------- Generators ----------------------------
####################################################################
# 这里与DRIT的G有一定区别，DRIT在运行时会不断地cat style code到每一步生成的结果中，使得第二维不断变大
# 目前不清楚这么做的意义, 可能是让每一层都附上style的影响？
# 目前先不考虑在每次操作中加入新的style
class Generator(nn.Module):
    def __init__(self, output_dim_x, output_dim_y, nz):
        super(Generator, self).__init__()
        self.nz = nz
        channel = 256 + self.nz

        decX1 = []
        decY1 = []
        for i in range(0, 3):
            decX1 += [INSResBlock(channel, channel)]
            decY1 += [INSResBlock(channel, channel)]
        self.decX1 = nn.Sequential(*decX1)
        self.decY1 = nn.Sequential(*decY1)

        self.decX2 = nn.Sequential(
            ReLUINSConvTranspose2d(channel, channel // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self.decY2 = nn.Sequential(
            ReLUINSConvTranspose2d(channel, channel // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        channel = channel//2

        self.decX3 = nn.Sequential(
            ReLUINSConvTranspose2d(channel, channel // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self.decY3 = nn.Sequential(
            ReLUINSConvTranspose2d(channel, channel // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        channel = channel//2

        self.decX4 = nn.Sequential(
            nn.ConvTranspose2d(channel, output_dim_x, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )
        self.decY4 = nn.Sequential(
            nn.ConvTranspose2d(channel, output_dim_y, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def forward_x(self, xands):
        out1 = self.decX1(xands)
        out2 = self.decX2(out1)
        out3 = self.decX3(out2)
        out4 = self.decX4(out3)
        return out4

    def forward_y(self, yands):
        out1 = self.decY1(yands)
        out2 = self.decY2(out1)
        out3 = self.decY3(out2)
        out4 = self.decY4(out3)
        return out4

class GlobalLevelResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dropout=0.0):
        super(GlobalLevelResBlock, self).__init__()
        self.model_x = nn.Sequential(
            INSResBlock(inplanes, planes)
        )
        self.model_y = nn.Sequential(
            INSResBlock(inplanes, planes)
        )

    def forward_x(self, x, s):
        out = self.model_x(x)
        s_img = s.view(s.size(0), s.size(1), 1, 1).expand(s.size(0), s.size(1), x.size(2), x.size(3))
        x_and_z = torch.cat([out, s_img], 1)
        return x_and_z

    def forward_y(self, y, s):
        out = self.model_y(y)
        s_img = s.view(s.size(0), s.size(1), 1, 1).expand(s.size(0), s.size(1), y.size(2), y.size(3))
        y_and_z = torch.cat([out, s_img], 1)
        return y_and_z

class InstanceLevelResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dropout=0.0):
        super(InstanceLevelResBlock, self).__init__()
        self.model_x = nn.Sequential(
            INSResBlock(inplanes, planes)
        )
        self.model_y = nn.Sequential(
            INSResBlock(inplanes, planes)
        )

    def forward_x(self, x, s):
        out = self.model_x(x)
        s_img = s.view(s.size(0), s.size(1), 1, 1).expand(s.size(0), s.size(1), x.size(2), x.size(3))
        x_and_z = torch.cat([out, s_img], 1)
        return x_and_z

    def forward_y(self, y, s):
        out = self.model_y(y)
        s_img = s.view(s.size(0), s.size(1), 1, 1).expand(s.size(0), s.size(1), y.size(2), y.size(3))
        y_and_z = torch.cat([out, s_img], 1)
        return y_and_z

# copied from cyclegan
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        model_1 = model.copy()

        self.model_x = nn.Sequential(*model)
        self.model_y = nn.Sequential(*model_1)

    def forward(self, input_x, input_y):
        out_x = self.model_x(input_x)
        out_y = self.model_y(input_y)
        return out_x, out_y

    def forward_x(self, input):
        """Standard forward"""
        return self.model_x(input)

    def forward_y(self, input):
        return self.model_y(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


####################################################################
#---------------------------  Detector  ----------------------------
####################################################################
class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        self.CLASSES = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
            'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

    def forward(self, images):
        detects = self.model(images)

        unloader = transforms.ToPILImage()
        # keep only predictions with 0.7+ confidence
        imgs = [unloader(images[i].cpu()) for i in range(images.size(0))]
        probas = detects['pred_logits'].softmax(-1)[:, :, :-1]
        keep = probas.max(-1).values > 0.9
        pros = [probas[i][keep[i]] for i in range(keep.size(0))]
        bboxes_scaled = [self.rescale_bboxes(detects['pred_boxes'][0, keep[i]], (images.size(-2), images.size(-1)))
                         for i in range(detects['pred_boxes'].size(0))]

        regions = []
        boxes = []
        for i in range(len(pros)):
            region = []
            box = []
            for pro, b in zip(pros[i], bboxes_scaled[i]):
                cl = pro.argmax()
                if self.CLASSES[cl] == 'person':
                    box_numpy = b.cpu().detach().numpy()
                    xl, yl, xr, yr = box_numpy
                    if abs(xr - xl) < 40 or abs(yr - yl) < 40:
                        continue
                    instance = imgs[i].crop(box_numpy)
                    region.append(instance)
                    box.append(b)
            regions.append(region)
            boxes.append(box)

        trans = [ToTensor()]
        trans = Compose(trans)
        regions = [[trans(reg).unsqueeze(0) for reg in region] for region in regions]
        return regions, boxes

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device='cuda')
        return b

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)


####################################################################
#-------------------------- Basic Blocks --------------------------
####################################################################
class LeakyReLUConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]

        if sn:
            model += [nn.utils.spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]

        if 'norm' == 'Instance':
            model += [nn.InstanceNorm2d(n_out, affine=False)]

        model += [nn.LeakyReLU(inplace=False)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)

class ReLUINSConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(ReLUINSConv2d, self).__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True),
            nn.InstanceNorm2d(n_out, affine=False),
            nn.ReLU(inplace=False)
        )
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)

class INSResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dropout=0.0):
        super(INSResBlock, self).__init__()
        model = []
        model += self.conv3x3(inplanes, planes, stride)
        model += [nn.InstanceNorm2d(planes)]
        model += [nn.ReLU(inplace=False)]
        model += self.conv3x3(planes, planes)
        model += [nn.InstanceNorm2d(planes)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def conv3x3(self, in_planes, out_planes, stride=1):
        return [nn.ReflectionPad2d(1), nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride)]

    def forward(self, x):
        residual = x
        out = self.model(x) + residual
        return out

class GaussianNoiseLayer(nn.Module):
    def __init__(self):
        super(GaussianNoiseLayer, self).__init__()

    def forward(self, x):
        if self.training == False:
            return x
        else:
            noise = Variable(torch.randn(x.size()).cuda(x.get_device()))
        return x + noise

class MisINSResBlock(nn.Module):
    def __init__(self, dim, dim_extra, stride=1, dropout=0.0):
        super(MisINSResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            self.conv3x3(dim, dim, stride),
            nn.InstanceNorm2d(dim)
        )
        self.conv2 = nn.Sequential(
            self.conv3x3(dim, dim, stride),
            nn.InstanceNorm2d(dim)
        )
        self.blk1 = nn.Sequential(
            self.conv1x1(dim + dim_extra, dim + dim_extra),
            nn.ReLU(inplace=False),
            self.conv1x1(dim + dim_extra, dim),
            nn.ReLU(inplace=False)
        )
        self.blk2 = nn.Sequential(
            self.conv1x1(dim + dim_extra, dim + dim_extra),
            nn.ReLU(inplace=False),
            self.conv1x1(dim + dim_extra, dim),
            nn.ReLU(inplace=False)
        )
        model = []
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.blk1.apply(gaussian_weights_init)
        self.blk2.apply(gaussian_weights_init)

    def forward(self, x, z):
        residual = x
        z_expand = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        o1 = self.conv1(x)
        o2 = self.blk1(torch.cat([o1, z_expand], dim=1))
        o3 = self.conv2(o2)
        out = self.blk2(torch.cat([o3, z_expand], dim=1)) + residual
        return out

    def conv3x3(self, dim_in, dim_out, stride=1):
        return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride))

    def conv1x1(self, dim_in, dim_out):
        return nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)

class ReLUINSConvTranspose2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
        super(ReLUINSConvTranspose2d, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True),
            LayerNorm(n_out),
            nn.ReLU(inplace=False)
        )
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)

class LayerNorm(nn.Module):
    def __init__(self, n_out, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.n_out = n_out
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
            self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))
        return

    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
            return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
        else:
            return F.layer_norm(x, normalized_shape)




####################################################################
#------------------------- Basic Functions -------------------------
####################################################################
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

def get_scheduler(optimizer, opts, cur_ep=-1):
    if opts.lr_policy == 'lambda':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
    elif opts.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
    else:
        return NotImplementedError('no such learn rate policy')
    return scheduler


