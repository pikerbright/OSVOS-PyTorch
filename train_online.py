# Package Includes
from __future__ import division

import os
import socket
import timeit
from datetime import datetime
from tensorboardX import SummaryWriter

# PyTorch includes
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

# Custom includes
from dataloaders import davis_2016 as db
from dataloaders import custom_transforms as tr
from util import visualize as viz
import scipy.misc as sm
import networks.vgg_osvos as vo
from layers.osvos_layers import class_balanced_cross_entropy_loss
from dataloaders.helpers import *
from mypath import Path

# Setting of parameters
if 'SEQ_NAME' not in os.environ.keys():
    seq_name = 'blackswan'
else:
    seq_name = str(os.environ['SEQ_NAME'])

db_root_dir = Path.db_root_dir()
save_dir = Path.save_root_dir()

if not os.path.exists(save_dir):
    os.makedirs(os.path.join(save_dir))

vis_net = 0  # Visualize the network?
vis_res = 0  # Visualize the results?
nAveGrad = 5  # Average the gradient every nAveGrad iterations
nEpochs = 2000 * nAveGrad  # Number of epochs for training
snapshot = nEpochs  # Store a model every snapshot epochs

# Parameters in p are used for the name of the model
p = {
    'trainBatch': 12,  # Number of Images in each mini-batch
    }
seed = 0

parentModelName = 'blackswan'
parentEpoch = 10000

# Select which GPU, -1 if CPU
# gpu_id = 0
# device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

def load_network(network):
    save_path = os.path.join(save_dir, parentModelName+'_epoch-'+str(parentEpoch-1)+'.pth')
    state_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        print("{}".format(k))
        namekey = k[7:] # remove `module.`
        new_state_dict[namekey] = v
    # load params
    network.load_state_dict(new_state_dict)
    return network

# Network definition
net = vo.OSVOS(pretrained=0)
# net.load_state_dict(torch.load(os.path.join(save_dir, parentModelName+'_epoch-'+str(parentEpoch-1)+'.pth'),
#                                map_location=lambda storage, loc: storage))
net = load_network(net)

# Logging into Tensorboard
log_dir = os.path.join(save_dir, 'runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname()+'-'+seq_name)
writer = SummaryWriter(log_dir=log_dir)

# net.to(device)  # PyTorch 0.4.0 style

# Visualize the network
if vis_net:
    x = torch.randn(1, 3, 480, 854)
    x.requires_grad_()
    x = x.to(device)
    y = net.forward(x)
    g = viz.make_dot(y, net.state_dict())
    g.view()

# Use the following optimizer
lr = 1e-8
wd = 0.0002
optimizer = optim.SGD([
    {'params': [pr[1] for pr in net.stages.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd},
    {'params': [pr[1] for pr in net.stages.named_parameters() if 'bias' in pr[0]], 'lr': lr * 2},
    {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd},
    {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'bias' in pr[0]], 'lr': lr*2},
    {'params': [pr[1] for pr in net.upscale.named_parameters() if 'weight' in pr[0]], 'lr': 0},
    {'params': [pr[1] for pr in net.upscale_.named_parameters() if 'weight' in pr[0]], 'lr': 0},
    {'params': net.fuse.weight, 'lr': lr/100, 'weight_decay': wd},
    {'params': net.fuse.bias, 'lr': 2*lr/100},
    ], lr=lr, momentum=0.9)

device_ids = [0, 1, 2]
net = torch.nn.DataParallel(net, device_ids=device_ids).cuda()

# Preparation of the data loaders
# Define augmentation transformations as a composition
composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                          tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                                          tr.ToTensor()])
# Training dataset and its iterator
db_train = db.DAVIS2016(train=True, db_root_dir=db_root_dir, transform=composed_transforms, seq_name=None)
trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=4)

# Testing dataset and its iterator
db_test = db.DAVIS2016(train=False, db_root_dir=db_root_dir, transform=tr.ToTensor(), seq_name=None)
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4)


num_img_tr = len(trainloader)
num_img_ts = len(testloader)
loss_tr = []
aveGrad = 0

print("Start of Online Training, sequence: " + seq_name)
start_time = timeit.default_timer()
# Main Training and Testing Loop
for epoch in range(0, nEpochs):
    # One training epoch
    running_loss_tr = 0
    np.random.seed(seed + epoch)

    net.train()

    for ii, sample_batched in enumerate(trainloader):

        inputs, gts = sample_batched['image'], sample_batched['gt']

        # Forward-Backward of the mini-batch
        inputs.requires_grad_()
        # inputs, gts = inputs.to(device), gts.to(device)
        gts = gts.cuda(async=True)
        input_var = torch.autograd.Variable(inputs)
        gts_var = torch.autograd.Variable(gts)

        outputs = net(input_var)

        # Compute the fuse loss
        loss = class_balanced_cross_entropy_loss(outputs[-1], gts_var, size_average=False)
        running_loss_tr += loss.item()  # PyTorch 0.4.0 style

        # Print stuff
        if epoch % (nEpochs//20) == (nEpochs//20 - 1):
            running_loss_tr /= num_img_tr
            loss_tr.append(running_loss_tr)

            print('[Epoch: %d, numImages: %5d]' % (epoch+1, ii + 1))
            print('Loss: %f' % running_loss_tr)
            writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)

        optimizer.zero_grad()

        # Backward the averaged gradient
        loss /= nAveGrad
        loss.backward()
        aveGrad += 1

        # Update the weights once in nAveGrad forward passes
        if aveGrad % nAveGrad == 0:
            writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
            optimizer.step()
            # optimizer.zero_grad()
            aveGrad = 0

    # Save the model
    if (epoch % snapshot) == snapshot - 1 and epoch != 0:
        torch.save(net.state_dict(), os.path.join(save_dir, seq_name + '_epoch-'+str(epoch) + '.pth'))

stop_time = timeit.default_timer()
print('Online training time: ' + str(stop_time - start_time))


# Testing Phase
if vis_res:
    import matplotlib.pyplot as plt
    plt.close("all")
    plt.ion()
    f, ax_arr = plt.subplots(1, 3)

save_dir_res = os.path.join(save_dir, 'Results', seq_name)
if not os.path.exists(save_dir_res):
    os.makedirs(save_dir_res)


print('Testing Network')
with torch.no_grad():  # PyTorch 0.4.0 style
    # Main Testing Loop
    for ii, sample_batched in enumerate(testloader):

        img, gt, fname = sample_batched['image'], sample_batched['gt'], sample_batched['fname']

        # Forward of the mini-batch
        # inputs, gts = img.to(device), gt.to(device)
        gt = gt.cuda(async=True)
        inputs = torch.autograd.Variable(img)
        gt = torch.autograd.Variable(gt)

        outputs = net.forward(inputs)

        for jj in range(int(inputs.size()[0])):
            pred = np.transpose(outputs[-1].cpu().data.numpy()[jj, :, :, :], (1, 2, 0))
            pred = 1 / (1 + np.exp(-pred))
            pred = np.squeeze(pred)

            # Save the result, attention to the index jj
            sm.imsave(os.path.join(save_dir_res, os.path.basename(fname[jj]) + '.png'), pred)

            if vis_res:
                img_ = np.transpose(img.numpy()[jj, :, :, :], (1, 2, 0))
                gt_ = np.transpose(gt.numpy()[jj, :, :, :], (1, 2, 0))
                gt_ = np.squeeze(gt)
                # Plot the particular example
                ax_arr[0].cla()
                ax_arr[1].cla()
                ax_arr[2].cla()
                ax_arr[0].set_title('Input Image')
                ax_arr[1].set_title('Ground Truth')
                ax_arr[2].set_title('Detection')
                ax_arr[0].imshow(im_normalize(img_))
                ax_arr[1].imshow(gt_)
                ax_arr[2].imshow(im_normalize(pred))
                plt.pause(0.001)

writer.close()
