import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.modules.losses.lpips import LPIPS
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQLPIPS_Ref(nn.Module):
    def __init__(self, disc_start, codebook1_weight=1.0, codebook2_weight=1.0, G_step=1, 
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0, reverse_weight=1.0,
                 style_weight=10.0,
                 use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook1_weight = codebook1_weight
        self.codebook2_weight = codebook2_weight
        self.reverse_weight = reverse_weight
        self.style_weight = style_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.G_step = G_step
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    
    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std
    
    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return F.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = self.calc_mean_std(input)
        target_mean, target_std = self.calc_mean_std(target)
        return F.mse_loss(input_mean, target_mean) + \
               F.mse_loss(input_std, target_std)

    def forward(self, 
                codebook1_loss, 
                inputs, reconstructions, 
                quant, 
                indices_ref, indices, 
                optimizer_idx, global_step, last_layer=None, cond=None, split="train", 
                mapped_reconstructions=None, 
                diff_identity=None, quant_identity=None):

        style_loss = self.calc_style_loss(reconstructions, inputs)
        if mapped_reconstructions is None:
            reverse_loss = self.calc_content_loss(reconstructions, quant)
        else:
            reverse_loss = self.calc_content_loss(mapped_reconstructions, quant)
            
        if quant_identity is not None:
            identity_loss = self.calc_content_loss(quant_identity, inputs)
        else:
            identity_loss = None
            
        
        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)
            
            d_weight = torch.tensor(self.discriminator_weight) 
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            aeloss = self.reverse_weight * reverse_loss + self.codebook1_weight * codebook1_loss.mean() + self.style_weight * style_loss
            if identity_loss is not None:
                aeloss = aeloss + self.reverse_weight * identity_loss
            if diff_identity is not None:
                aeloss = aeloss + self.codebook1_weight * diff_identity.mean()
            loss = aeloss + d_weight * disc_factor * g_loss 

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_x2y_loss".format(split): codebook1_loss.detach().mean(),
                   "{}/reverse_loss".format(split): reverse_loss.detach().mean(),
                   "{}/style_loss".format(split): style_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            if identity_loss is not None:
                log["{}/identity_loss".format(split)] = identity_loss.detach().mean()
            if diff_identity is not None:
                log["{}/diff_identity".format(split)] = diff_identity.detach().mean()
            return loss, aeloss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            if not global_step % self.G_step == 0:
                disc_factor = disc_factor * 0.0
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
