import torch
import torch.nn as nn
from torch.nn import init
import random
import os
from torch.optim import lr_scheduler
import time
from datetime import date

class TemporalModelBranched:
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    def __init__(self):

        self.save_dir = os.path.abspath("/s/chopin/k/grad/sarmst/CR/checkpoints")  # save all the checkpoints to save_dir
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.model_names = ['G', 'D']
        self.netG = self.define_G()
        self.netD = self.define_D()
        self.optimizers = []
        self.image_paths = []
        self.criterionGAN = GANLoss()
        self.criterionL1 = torch.nn.L1Loss()
        self.epoch_count = 1
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizer_G = torch.optim.Adagrad(self.netG.parameters())
        self.optimizer_D = torch.optim.Adagrad(self.netD.parameters())
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        # self.schedulers = [self.get_scheduler(optimizer) for optimizer in self.optimizers]
        # self.metric = 0

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_A_0 = input['A_0']
        self.real_A_1 = input['A_1']
        self.real_A_2 = input['A_2']
        options = [self.real_A_0.clone(), self.real_A_1.clone(), self.real_A_2.clone()]
        scramble = random.sample(range(0,3), 3)
        self.real_A_0 = options[scramble[0]]
        self.real_A_1 = options[scramble[1]]
        self.real_A_2 = options[scramble[2]]
        self.real_A = torch.cat((self.real_A_0, self.real_A_1, self.real_A_2), 1)
        self.real_A_input = options
        self.real_B = input['B']
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A_input)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * 100.0
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    # def optimize_parameters(self):
    #     self.forward()                   # compute fake images: G(A)
    #     # update D
    #     self.set_requires_grad(self.netD, True)  # enable backprop for D
    #     self.optimizer_D.zero_grad()     # set D's gradients to zero
    #     self.backward_D()                # calculate gradients for D
    #     self.average_gradients(self)
    #     self.optimizer_D.step()          # update D's weights
    #     # update G
    #     self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
    #     self.optimizer_G.zero_grad()        # set G's gradients to zero
    #     self.backward_G()                   # calculate graidents for G
    #     self.average_gradients(self)
    #     self.optimizer_G.step()             # udpate G's weights


    def define_G(self):
        return ResnetGenerator()

    def define_D(self):
        return NLayerDiscriminator()

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step(self.metric)
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def save_networks(self, epoch):

        save_filename_netD = '%s_net_%s_%s.pth' % (epoch, "netD", str(int(round(time.time() * 1000))))
        save_filename_netG = '%s_net_%s_%s.pth' % (epoch, "netG", str(int(round(time.time() * 1000))))
        # save_filename = '%s_net_%s_%s.pth' % (epoch, name, str(int(round(time.time() * 1000))))
        time.sleep(10)
        save_path_netD = os.path.join("/s/chopin/k/grad/sarmst/CR/savedModels", "netD")
        save_path_netG = os.path.join("/s/chopin/k/grad/sarmst/CR/savedModels", "netG")

        torch.save(self.netD.cpu().state_dict(), save_path_netD)
        torch.save(self.netG.cpu().state_dict(), save_path_netG)

    def get_scheduler(self, optimizer):
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + self.epoch_count - 100) / float(101)
                return lr_l
            return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        torch.manual_seed(18)
        gan_mode = "vanilla"
        target_real_label=1.0
        target_fake_label=0.0


        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """
    def __init__(self):

        torch.manual_seed(18)
        super(ResnetGenerator, self).__init__()
        model_initial = [nn.ReflectionPad2d(3),
                            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), bias=False),
                            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.ReLU(True)]

        model_intermediate = [nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True),
                                ResnetBlock(),
                                ResnetBlock(),
                                ResnetBlock(),
                                ResnetBlock(),
                                ResnetBlock(),
                                ResnetBlock(),
                                ResnetBlock(),
                                ResnetBlock(),
                                ResnetBlock()]

        model_final = [nn.ConvTranspose2d(3072, 1536, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False),
                        nn.BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(1536, 768, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False),
                        nn.BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(768, 384, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False),
                        nn.BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(3),
                        nn.Conv2d(384, 3, kernel_size=(7, 7), stride=(1, 1)),
                        nn.Tanh()]

        self.model_initial = nn.Sequential(*model_initial)
        self.model_intermediate = nn.Sequential(*model_intermediate)
        self.model_final = nn.Sequential(*model_final)

    def forward(self, input):
        """Standard forward"""
        input_0 = input[0]
        input_1 = input[1]
        input_2 = input[2]
        output_0 = self.model_initial(input_0)
        output_1 = self.model_initial(input_1)
        output_2 = self.model_initial(input_2)
        intermediate_input_01 = torch.cat((output_0, output_1), 1)
        intermediate_input_02 = torch.cat((output_0, output_2), 1)
        intermediate_input_12 = torch.cat((output_1, output_2), 1)
        output_intermediate_0 = self.model_intermediate(intermediate_input_01)
        output_intermediate_1 = self.model_intermediate(intermediate_input_02)
        output_intermediate_2 = self.model_intermediate(intermediate_input_12)
        return self.model_final(torch.cat((output_intermediate_0, output_intermediate_1, output_intermediate_2), 1))

class ResnetBlock(nn.Module):

    def __init__(self):
        super(ResnetBlock, self).__init__()
        torch.manual_seed(18)
        self.conv_block = self.build_conv_block()

    def build_conv_block(self):
        torch.manual_seed(18)
        return nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class NLayerDiscriminator(nn.Module):
    def __init__(self):
        super(NLayerDiscriminator, self).__init__()
        torch.manual_seed(18)
        self.model = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)))


    def forward(self, input):
        """Standard forward."""
        return self.model(input)
