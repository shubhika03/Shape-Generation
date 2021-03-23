import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import tqdm
import numpy as np
import os

import model
import dataset

def get_args():
    parser = argparse.ArgumentParser(description="Training options")
    parser.add_argument("--data_path", type=str, default="./coefficients.npy",
                        help="Path to the root data directory")
    parser.add_argument("--save_path", type=str, default="./models/",
                        help="Path to save models")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Mini-Batch size")
    parser.add_argument("--lr", type=float, default=2.5e-3,
                        help="learning rate for generator")
    parser.add_argument("--lr_D", type=float, default=1e-4,
                        help="discriminator learning rate")
    return parser.parse_args()

def make_some_noise(batch_size):
    return torch.rand(batch_size, 100)

class Trainer:
    def __init__(self):
        self.opt = get_args()

        self.parameters_to_train = []
        self.parameters_to_train_D = []
        self.criterion_d = nn.BCELoss()

        self.models = {}
        self.models['generator'] = model.Generator()
        self.models['discriminator'] = model.Discriminator()

        self.parameters_to_train += list(self.models['generator'].parameters())
        self.parameters_to_train_D += list(self.models['discriminator'].parameters())

        self.model_optimizer = optim.Adam(
            self.parameters_to_train, self.opt.lr)

        self.model_optimizer_D = optim.Adam(
            self.parameters_to_train_D, self.opt.lr_D)

        self.valid = Variable(
            torch.Tensor(
                np.ones(
                    (self.opt.batch_size,
                     1))),
            requires_grad=False).float()

        self.fake = Variable(
            torch.Tensor(
                np.zeros(
                    (self.opt.batch_size,
                     1))),
            requires_grad=False).float()

        train_dataset = dataset.Coefficient_dataset(self.opt.data_path)

        self.train_loader = DataLoader(
            train_dataset,
            self.opt.batch_size,
            True,
            drop_last=True)

    def train(self):

        for self.epoch in range(100):
            print("epoch number is " + str(self.epoch))
            for batch_idx, true_coeff in tqdm.tqdm(enumerate(self.train_loader)):

                random_input = make_some_noise(self.opt.batch_size)
                fake_coeff = self.models['generator'](random_input)
                
                real_pred = self.models['discriminator'](true_coeff)
                

                #training discriminator
                fake_pred = self.models['discriminator'](fake_coeff.detach())
                loss_d = self.criterion_d(fake_pred, self.fake) + self.criterion_d(real_pred, self.valid)
                self.model_optimizer_D.zero_grad()
                loss_d.backward()
                self.model_optimizer_D.step()
                print("discriminator loss is " + str(loss_d.item()))

                #training generator
                fake_stats = self.models['discriminator'](fake_coeff, intermediate=True)
                true_stats = self.models['discriminator'](true_coeff, intermediate=True)
                # fake_stats = fake_coeff
                # true_stats = true_coeff
                loss_g1 = torch.norm(torch.mean(fake_stats, dim=0) - torch.mean(true_stats, dim=0))
                cov1 = torch.mean(torch.pow(fake_stats - torch.mean(fake_stats, dim=0), 2), dim=0)
                cov2 = torch.mean(torch.pow(true_stats - torch.mean(true_stats, dim=0), 2), dim=0)
                loss_g2 = torch.norm(cov1-cov2)
                loss_g = loss_g1 + loss_g2
                self.model_optimizer.zero_grad()
                loss_g.backward()
                self.model_optimizer.step()

                
                print("generator loss is " + str(loss_g.item()))

            self.save_model()

    def save_model(self):
        save_path = os.path.join(
            self.opt.save_path,
            "exp1",
            "weights_{}".format(
                self.epoch))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for model_name, model in self.models.items():
            model_path = os.path.join(save_path, "{}.pth".format(model_name))
            state_dict = model.state_dict()
            torch.save(state_dict, model_path)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()



