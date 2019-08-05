import torch
from .base_model import BaseModel
from . import affineGAN_networks as networks


class AffineGANModel(BaseModel):
    def name(self):
        return "AffineGANModel"

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # AffineGAN use instance norm
        parser.set_defaults(pool_size=0, no_lsgan=True, norm="instance")
        parser.set_defaults(dataset_mode="affineGAN")
        parser.set_defaults(netG="unet_256")
        if is_train:
            parser.add_argument(
                "--lambda_L1", type=float, default=100.0, help="weight for L1 loss"
            )

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = [
            "G_GAN_D1",
            "G_L1",
            "D_real",
            "D_fake",
            "G_GAN_D_alpha",
            "D_alpha",
            "img_recons",
        ]
        if not opt.no_patch:
            self.loss_names += [
                "G_GAN_patch",
                "D_real_patch",
                "D_fake_patch",
                "D_patch",
            ]

        if self.isTrain:
            self.visual_names = ["input_A", "fake_B", "real_B"]
            self.model_names = ["G", "D", "D_alpha"]
            if not opt.no_patch:
                self.model_names.append("D_Patch")

        else:  # during test time, only load Gs
            self.visual_names = ["input_A"] + ["fake_B_list"]
            self.model_names = ["G"]
        # load/define networks
        self.netG = networks.define_G(
            opt.input_nc,
            opt.input_nc,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(
                opt.input_nc + opt.output_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                use_sigmoid,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )
            self.netD_alpha = networks.define_D_alpha(
                opt.train_imagenum,
                opt.norm,
                use_sigmoid,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(
                self.device
            )
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D_Alpha = torch.optim.Adam(
                self.netD_alpha.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_D_Alpha)

            if not opt.no_patch:
                self.netD_Patch = networks.define_D(
                    opt.input_nc + opt.output_nc,
                    opt.ndf,
                    opt.netD,
                    opt.n_layers_D,
                    opt.norm,
                    use_sigmoid,
                    opt.init_type,
                    opt.init_gain,
                    self.gpu_ids,
                )
                self.optimizer_D_Patch = torch.optim.Adam(
                    self.netD_Patch.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
                )
                self.optimizers.append(self.optimizer_D_Patch)

    def get_alpha(self, f_t0, f_t, f_t0_res):
        return torch.abs(torch.sum((f_t - f_t0) * f_t0_res)) / (
            f_t0_res.norm() + 1e-6
        )

    def set_input(self, input):
        self.input_A = input["A"].to(self.device)
        self.image_paths = input["A_paths"]
        if self.isTrain:
            self.input_B_list = []
            self.input_B_patch_list = []
            for img_idx in range(self.opt.train_imagenum):
                self.input_B_list.append(input["B_list"][img_idx].to(self.device))
                if not self.opt.no_patch:
                    self.input_A_patch = input["A_patch"].to(self.device)
                    self.input_B_patch_list.append(
                        input["B_patch_list"][img_idx].to(self.device)
                    )

    def forward(self):
        if not self.opt.no_patch and self.isTrain:
            self.input_A_img_patch = self.input_A * self.input_A_patch

        self.t0_reconstruct, f_t0 = self.netG(self.input_A, 1.0, 0.0, self.isTrain)

        self.real_B_list = []
        self.fake_B_list = []
        self.B_reconstruct_img_list = []
        if not self.opt.no_patch:
            self.fake_B_img_patch_list = []
            self.real_B_img_patch_list = []

        alpha_list_torch = []

        _, f_t0_res = self.netG(self.input_A, 0.0, 1.0, self.isTrain)
        f_t0_res = torch.squeeze(f_t0_res)
        f_t0 = torch.squeeze(f_t0)

        for img_idx in range(self.opt.train_imagenum):
            real_B = self.input_B_list[img_idx]

            t_reconstruct, f_t = self.netG(real_B, 1.0, 0.0, self.isTrain)
            self.B_reconstruct_img_list.append(t_reconstruct)
            f_t = torch.squeeze(f_t)
            alpha = self.get_alpha(f_t0, f_t, f_t0_res)
            alpha_list_torch.append(alpha.view(1))

            fake_B, _ = self.netG(self.input_A, 1.0, float(alpha), self.isTrain)

            self.real_B_list.append(real_B)
            self.fake_B_list.append(fake_B)

            if not self.opt.no_patch:
                real_B_patch = self.input_B_patch_list[img_idx]
                real_B_img_patch = real_B * real_B_patch
                fake_B_img_patch = fake_B * real_B_patch
                self.fake_B_img_patch_list.append(fake_B_img_patch)
                self.real_B_img_patch_list.append(real_B_img_patch)

        self.alpha_list_torch = torch.stack(alpha_list_torch, dim=1)

        self.alpha_list_sample = torch.rand(1, self.opt.train_imagenum).to(self.device)

        self.fake_B = self.fake_B_list[0]
        self.real_B = self.real_B_list[0]

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        loss_D = 0
        for img_idx in range(self.opt.train_imagenum):
            fake_AB = torch.cat((self.input_A, self.fake_B_list[img_idx]), 1)
            pred_fake = self.netD(fake_AB.detach())
            self.loss_D_fake = self.criterionGAN(pred_fake, False)

            # Real
            real_AB = torch.cat((self.input_A, self.real_B_list[img_idx]), 1)
            pred_real = self.netD(real_AB.detach())
            self.loss_D_real = self.criterionGAN(pred_real, True)

            # Combined loss
            loss_D += (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D = loss_D / (self.opt.train_imagenum + 0.0)
        self.loss_D.backward()

    def backward_D_patch(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        loss_D_patch = 0
        for img_idx in range(self.opt.train_imagenum):
            fake_AB_patch = torch.cat(
                (self.input_A_img_patch, self.fake_B_img_patch_list[img_idx]), 1
            )
            pred_fake_patch = self.netD_Patch(fake_AB_patch.detach())
            self.loss_D_fake_patch = self.criterionGAN(pred_fake_patch, False)

            # Real
            real_AB_patch = torch.cat(
                (self.input_A_img_patch, self.real_B_img_patch_list[img_idx]), 1
            )
            pred_real_patch = self.netD_Patch(real_AB_patch.detach())
            self.loss_D_real_patch = self.criterionGAN(pred_real_patch, True)

            # Combined loss
            loss_D_patch += (self.loss_D_fake_patch + self.loss_D_real_patch) * 0.5
        self.loss_D_patch = loss_D_patch / (self.opt.train_imagenum + 0.0)
        self.loss_D_patch.backward()

    def backward_D_alpha(self):
        # Fake
        # stop backprop to the generator by detaching fake_B

        pred_fake_alpha = self.netD_alpha(self.alpha_list_torch.detach())
        pred_true_alpha = self.netD_alpha(self.alpha_list_sample.detach())

        self.loss_D_fake_alpha = self.criterionGAN(pred_fake_alpha, False)
        self.loss_D_real_alpha = self.criterionGAN(pred_true_alpha, True)
        # Combined loss
        self.loss_D_alpha = (
            (self.loss_D_fake_alpha + self.loss_D_real_alpha) * 0.5 * self.opt.lambda_A
        )
        self.loss_D_alpha.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        loss_G = 0
        loss_G_GAN_D1 = 0
        loss_G_GAN_patch = 0
        loss_G_L1 = 0
        img_recons_loss = 0

        pred_fake_alpha = self.netD_alpha(self.alpha_list_torch)
        loss_G_GAN_D_alpha = (
            self.criterionGAN(pred_fake_alpha, True) * self.opt.lambda_A
        )

        for img_idx in range(self.opt.train_imagenum):
            fake_AB = torch.cat((self.input_A, self.fake_B_list[img_idx]), 1)
            pred_fake = self.netD(fake_AB)
            current_loss_G_GAN_D1 = self.criterionGAN(pred_fake, True)
            loss_G_GAN_D1 += current_loss_G_GAN_D1 / (self.opt.train_imagenum + 0.0)

            # First_2, G(A) should fake the discriminator_patch
            if not self.opt.no_patch:
                fake_AB_patch = torch.cat(
                    (self.input_A_img_patch, self.fake_B_img_patch_list[img_idx]), 1
                )
                pred_fake_patch = self.netD_Patch(fake_AB_patch)
                current_loss_G_GAN_patch = self.criterionGAN(pred_fake_patch, True)
                loss_G_GAN_patch += current_loss_G_GAN_patch / (
                    self.opt.train_imagenum + 0.0
                )
                loss_G += current_loss_G_GAN_patch

            # Second, G(A) = B
            current_loss_G_L1 = (
                self.criterionL1(self.fake_B_list[img_idx], self.real_B_list[img_idx])
                * self.opt.lambda_L1
            )
            loss_G_L1 += current_loss_G_L1 / (self.opt.train_imagenum + 0.0)
            current_img_recons_loss = (
                self.criterionL1(
                    self.B_reconstruct_img_list[img_idx], self.real_B_list[img_idx]
                )
                * 10.0
            )
            img_recons_loss = current_img_recons_loss / (self.opt.train_imagenum + 0.0)
            loss_G += (
                current_loss_G_GAN_D1 + current_loss_G_L1 + current_img_recons_loss
            )

        loss_G += loss_G_GAN_D_alpha
        loss_G /= self.opt.train_imagenum + 0.0
        self.loss_G = loss_G
        self.loss_G_GAN_D1 = loss_G_GAN_D1
        self.loss_G_GAN_patch = loss_G_GAN_patch
        self.loss_G_L1 = loss_G_L1
        self.loss_G_GAN_D_alpha = loss_G_GAN_D_alpha
        self.loss_img_recons = img_recons_loss
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad(self.netD_alpha, True)
        self.optimizer_D_Alpha.zero_grad()
        self.backward_D_alpha()
        self.optimizer_D_Alpha.step()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        if not self.opt.no_patch:
            self.set_requires_grad(self.netD_Patch, True)
            self.optimizer_D_Patch.zero_grad()
            self.backward_D_patch()
            self.optimizer_D_Patch.step()
            self.set_requires_grad(self.netD_Patch, False)

        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netD_alpha, False)

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.fake_B_list = []
            for i in range(int(1.0 / self.opt.interval)):
                self.fake_B_list.append(
                    self.netG(self.input_A, 1.0, self.opt.interval * i, self.isTrain)[0]
                )

