import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from torchvision import  transforms as T
from skimage.color import ycbcr2rgb
from skimage.io import imsave
from skimage import img_as_ubyte

def write_figures(location, train_losses, val_losses):
    plt.plot(train_losses, label='training loss')
    plt.plot(val_losses, label='validation loss')

    plt.legend()
    plt.savefig(location + '/loss.png')
    plt.close('all')


def write_log(location, epoch, train_loss, val_loss):
    if epoch == 0:
        f = open(location + '/log.txt', 'w+')
        f.write('epoch\t\ttrain_loss\t\tval_loss\n')
    else:
        f = open(location + '/log.txt', 'a+')

    f.write(str(epoch) + '\t' + str(train_loss) + '\t' + str(val_loss) + '\n')

    f.close()

def savefig(epoch,  outputs, filename, postfix_name):
    test_mean, test_std = torch.tensor([0.5 ,0.5 ,0.5]), torch.tensor([0.5 ,0.5, 0.5])
    if epoch %1 == 0:
        prediction = outputs * test_std.view(3,1,1) + test_mean.view(3,1,1)
        prediction = prediction * 255

        #ycbcr to rgb
        prediction = ycbcr2rgb(prediction[0].detach().numpy().transpose(1,2,0))
        prediction = np.clip(prediction, -1, 1)
        imsave(".\\result\\%s_%s.png" % (filename[0].replace(' ','').split('\\')[-2], postfix_name), img_as_ubyte(prediction))
