# Reconstruction + KL divergence losses3D summed over all elements and batch
import torch
import torch.functional as F


# TODO test and class
def loss_vae(recon_x, x, mu, logvar, type="BCE", h1=0.1, h2=0.1):
    """
    see Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param recon_x:
    :param x:
    :param mu,logvar: VAE parameters
    :param type: choices BCE,L1,L2
    :param h1: reconsrtruction hyperparam
    :param h2: KL div hyperparam
    :return: total loss of VAE
    """
    batch = recon_x.shape[0]
    assert recon_x.size() == x.size()
    assert recon_x.shape[0] == x.shape[0]
    rec_flat = recon_x.view(batch, -1)
    x_flat = x.view(batch, -1)
    if type == "BCE":
        loss_rec = F.binary_cross_entropy(rec_flat, x_flat, reduction='sum')
    elif type == "L1":
        loss_rec = torch.sum(torch.abs(rec_flat - x_flat))
    elif type == "L2":
        loss_rec = torch.sum(torch.sqrt(rec_flat * rec_flat - x_flat * x_flat))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return loss_rec * h1 + KLD * h2
