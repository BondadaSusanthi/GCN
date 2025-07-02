import torch
import torch.nn as nn
import torch.optim
from util import AveragePrecisionMeter


class Engine(object):
    def __init__(self):
        self.ap_meter = AveragePrecisionMeter()

    def train(self, train_loader, model, criterion, optimizer, epoch):
        model.train()
        for i, (input, target) in enumerate(train_loader):
            input, target = input.float(), target.float()

            optimizer.zero_grad()
            output = model(input, self.label_embed)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            self.ap_meter.add(output.detach(), target)

            if i % 10 == 0:
                print(f"Epoch [{epoch}] Iter [{i}] Loss: {loss.item():.4f}")

        OP, OR, OF1, CP, CR, CF1 = self.ap_meter.overall()
        print(f"Train OP: {OP:.4f} OR: {OR:.4f} OF1: {OF1:.4f} CP: {CP:.4f} CR: {CR:.4f} CF1: {CF1:.4f}")
        self.ap_meter.reset()

    def validate(self, val_loader, model, criterion):
        model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                input, target = input.float(), target.float()

                output = model(input, self.label_embed)
                self.ap_meter.add(output, target)

        OP, OR, OF1, CP, CR, CF1 = self.ap_meter.overall()
        print(f"Val OP: {OP:.4f} OR: {OR:.4f} OF1: {OF1:.4f} CP: {CP:.4f} CR: {CR:.4f} CF1: {CF1:.4f}")
        self.ap_meter.reset()
