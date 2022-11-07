"""
This file generates accumulative adversarial examples for the training of ATTA
"""
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random

def perturb(model, cand_model, x_nat, x_adv, y, step_size=0.007, epsilon=0.031,
                 num_steps=10, random_start = False, l2 = False, attack=True, celoss = "Cross", bound=True):
    model.eval()

    if random_start: # done manually in ATTA training code
        if not l2:
            perturb = torch.FloatTensor(*x_adv.shape).uniform_(-epsilon, epsilon).to(x_adv.device)
            x_adv = x_adv + perturb
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            """"
                As seen in Foolbox
                References:
                .. [#Voel17] Voelker et al., 2017, Efficiently sampling vectors and coordinates
                    from the n-sphere and n-ball
                    http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
            """
            batch_size, n = x_adv.view(x_adv.size()[0], -1).size()

            rand_samples = torch.randn((batch_size, n + 1)).to(x_adv.device)
            norms = torch.linalg.norm(rand_samples, ord=2, dim=1, keepdim=True)
            rand_direction = (rand_samples / norms)[:, :n]

            x_adv = x_adv + epsilon * rand_direction.reshape(x_adv.size())
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv.requires_grad_()
    batch_size = len(x_nat)

    if celoss == 'Cross':
        ce_loss = nn.CrossEntropyLoss()
    elif celoss == "BCE":
        bce_loss = nn.BCELoss()
    elif celoss == "combined":
        ce_loss = nn.CrossEntropyLoss()
        bce_loss = nn.BCELoss()

    for i in range(num_steps):
        x_adv.requires_grad_()

        with torch.enable_grad():
            cifar_norm = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
            # loss = ce_loss(model(cifar_norm(x_adv)), y)
            # loss = ce_loss(model(x_adv), y)
            if attack:
                if celoss == "combined":
                    # push away from one meaning id
                    d_labels = torch.unsqueeze(torch.ones(y.size()[0], dtype=torch.float), dim=1).cuda()
                    loss = ce_loss(model(x_adv), y) + bce_loss(cand_model(x_adv), d_labels)
                elif celoss == "Cross":
                    loss = ce_loss(model(x_adv), y)
                elif celoss == "BCE":
                    d_labels = torch.unsqueeze(torch.zeros(y.size()[0], dtype=torch.float), dim=1).cuda()
                    loss = bce_loss(cand_model(x_adv), d_labels)
            else:
                loss = ce_loss(model(x_adv), torch.tensor([10] * y.size(dim=0)).to(y.device))

        grad = torch.autograd.grad(loss, x_adv)[0]

        if not l2:
            #modift to gradient descent
            x_adv = x_adv.detach() + step_size * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x_nat - epsilon), 
                              x_nat + epsilon)
        else:
            flat = grad.view(grad.size()[0], -1)
            factor = torch.linalg.norm(flat, ord=2, dim=1)
            ones = torch.ones(factor.size()).to(x_adv.device) * 1e-9
            factor = torch.where(factor > 0, factor, ones)
            grad = (flat / factor.unsqueeze(1)).view(grad.size())
            x_adv = x_adv.detach() + step_size * grad

            flat_diff = (x_adv - x_nat).view(x_adv.size()[0], -1)
            factor = torch.linalg.norm(flat_diff, ord=2, dim=1)
            factor = torch.where(factor > 0, factor, ones)
            eps_vector = torch.ones(factor.size()).to(x_adv.device) * epsilon
            factor = eps_vector / factor
            factor = torch.where(factor > torch.ones(factor.size()).to(x_adv.device), torch.ones(factor.size()).to(x_adv.device), factor)
            pert = (factor.unsqueeze(1) * flat_diff).view(x_adv.size())
            x_adv = x_nat + pert

        if bound:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv