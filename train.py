from torch.autograd import Variable
import torch.autograd as autograd
import torch
import torch.optim as optim
from helper import write_log, write_figures, savefig
from dataset import get_loader
import torch.nn as nn
from model import simple_model, init_net
from tqdm import tqdm
from torchvision import transforms as T
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import numpy as np


class cos_sim(torch.nn.Module):

    def __init__(self):
        super(cos_sim, self).__init__()
        # self.win_size = win_size
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, gt, out):
        return 1 - torch.mean(self.cos(gt, out))


def ycrcb_to_rgb_torch(input_tensor, delta=0.5):
    y, cr, cb = input_tensor[:, 0, :, :], input_tensor[
        :, 1, :, :], input_tensor[:, 2, :, :]
    r = torch.unsqueeze(y + 1.403 * (cr - delta), 1)
    g = torch.unsqueeze(y - 0.714 * (cr - delta) - 0.344 * (cb - delta), 1)
    b = torch.unsqueeze(y + 1.773 * (cb - delta), 1)

    return torch.cat([r, g, b], 1)


class TVLoss(nn.Module):

    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def fit(epoch, model,  optimizer, criterion,  ssim, criterionMSE,  device, data_loader, phase='training'):
    if phase == 'training':
        model.train()
    else:
        model.eval()
    show = 0
    running_loss = 0
    total_loss_d = 0
    count = 0
    running_Q = 0
    running_S = 0
    running_N = 0

    for low, high, groundtruth, filename in tqdm(data_loader):
        # inputs = out[:,0:1,:,:].to(device)
        groundtruth = groundtruth.to(device)
        targets_y = groundtruth[:, 0:1, :, :]
        low_y = low[:, 0:1, :, :].to(device)
        high_y = high[:, 0:1, :, :].to(device)

        targets_cb = groundtruth[:, 1:2, :, :].to(device)
        targets_cr = groundtruth[:, 2:3, :, :].to(device)
        # print(targets_cbcr.shape)
        low_cb = low[:, 1:2, :, :].to(device)
        high_cb = high[:, 1:2, :, :].to(device)
        low_cr = low[:, 2:3, :, :].to(device)
        high_cr = high[:, 2:3, :, :].to(device)

        if phase == 'predict':
            model.eval()
            print(r'load model')
            model.load_state_dict(torch.load(
                'output/weight_best.pth', map_location=device))

        if phase == 'training':
            optimizer.zero_grad()

        else:
            model.eval()

        y, cb, cr = model(low_y, high_y, low_cb, high_cb, low_cr, high_cr)

        lossL1_y = criterion(y, targets_y)
        lossL1_cb = criterion(cb, targets_cb)
        lossL1_cr = criterion(cr, targets_cr)
        lossL1 = lossL1_y + lossL1_cb + lossL1_cr
        output = torch.cat([y, cb, cr], 1)
        output_rgb = ycrcb_to_rgb_torch(output)
        groundtruth_rgb = ycrcb_to_rgb_torch(groundtruth)

        groundtruth_msssim = (output_rgb + 1) / 2  # [-1, 1] => [0, 1]
        outputs_msssim = (groundtruth_rgb + 1) / 2  # [-1, 1] => [0, 1]

        lossm = ssim(outputs_msssim, groundtruth_msssim)

        # look have a fog
        loss = lossL1 * 0.6 + lossm * 0.2

        print(' lossL1: %.4f lossm: %.4f' %
              (lossL1.item(), lossm.item()))
        #+ lossMSSIM
        running_loss += loss.item()
        if phase == 'training':
            show = 0
            loss.backward()
            optimizer.step()

        # print only 20 img to check result
        elif phase == 'validation' and show <= 20:
            savefig(epoch, output.cpu().detach(), filename, 'simple')
            show += 1
        elif phase == 'predict':
            savefig(epoch, output.cpu().detach(), filename, 'fused')

    epoch_loss = running_loss / len(data_loader.dataset)
    return epoch_loss


def train(root, device, model, epochs, bs, lr):
    # print('start training ...........')
    train_loader, val_loader = get_loader(
        root=root, batch_size=bs, shuffle=True)

    criterion = nn.L1Loss().to(device)
    criterionMSE = nn.MSELoss().to(device)
    ssim = MS_SSIM()
    train_losses, val_losses, total_Q, total_S, total_N = [
    ], [], [], [], []

    for epoch in range(epochs):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_epoch_loss = fit(
            epoch, model, optimizer, criterion,  ssim, criterionMSE,  device, train_loader, phase='training')
        val_epoch_loss = fit(
            epoch, model, optimizer, criterion,   ssim,
            criterionMSE,   device, val_loader, phase='validation')

        # phase='predict'
        # val_epoch_loss = fit(
        # epoch, model, optimizer, criterion,   ssim,
        # criterionMSE,   device, val_loader, phase='predict')

        print('-----------------------------------------')

        if epoch % 10 == 0 and epoch != 0:
            torch.save({'model_state_dict': model.state_dict(), 'refunet_state_dict': refunet.state_dict()},
                       'output/weight_{}.pth'.format(epoch))

        if epoch == 0 or val_epoch_loss <= np.min(val_losses):
            torch.save({'model_state_dict': model.state_dict(), 'refunet_state_dict': refunet.state_dict()},
                       'output/weight_best.pth'.format(epoch))

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)

        write_figures('output', train_losses, val_losses)
        write_log('output', epoch, train_epoch_loss, val_epoch_loss)


if __name__ == '__main__':
    # device = torch.device('cpu')
    device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
    model = simple_model().to(device)

    init_net(model)
    batch_size = 1
    num_epochs = 200
    learning_rate = 0.001
    root = 'data/train'
    train(root, device, model,
          num_epochs, batch_size, learning_rate)
