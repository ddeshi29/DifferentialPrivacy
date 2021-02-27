import time
from misc_functions import (
                            convert_to_grayscale,
                            preprocess_image,
                            # get_example_params,
                            # save_image,
                            # save_gradient_images
                            )

import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from opacus import PrivacyEngine
from PIL import Image, ImageFilter
from opacus.utils import module_modification
from opacus.dp_model_inspector import DPModelInspector


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def accuracy(preds, labels):
    return (preds == labels).mean()

def train(model, train_loader, optimizer, epoch, device,dp):
    virtual_batch_rate = int(VIRTUAL_BATCH_SIZE / BATCH_SIZE)
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    for i, (images, target) in enumerate(train_loader):        
        images = images.to(device)
        target = target.to(device)

        # optimizer.zero_grad()
        # compute output
        output = model(images)
        loss = criterion(output, target)
        
        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()
        
        # measure accuracy and record loss
        acc = accuracy(preds, labels)
        # acc = criterion(preds, labels)

        losses.append(loss.item())
        top1_acc.append(acc)
        
        loss.backward()
        if dp:
            if ((i + 1) % virtual_batch_rate == 0) or ((i + 1) == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
            else:
                optimizer.virtual_step() # take a virtual step
        else:
            optimizer.step()
            optimizer.zero_grad()
        if i % 200 == 0:
            if dp:
                epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(DELTA)
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {DELTA})"
                )
            else:
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                )
    


# function used to train a model 
# net           - trained neural net to test
# trainset      - set of training images
# trainloader   - dataloader loaded with images images in trainset
# device        - device to use to forward test image through network
# dp            - are we training with differential privacy?

MAX_GRAD_NORM = 1.2
NOISE_MULTIPLIER = 1.0
DELTA = 1e-3
LR = 9e-4
NUM_WORKERS = 2
#NonDP 
# BATCH_SIZE = 64
# VIRTUAL_BATCH_SIZE = 256

#DP
BATCH_SIZE = 8
VIRTUAL_BATCH_SIZE = 16
def train_model(net,trainloader,trainset,device,dp):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(net.parameters(), lr=LR)
    # optimizer = torch.optim.SGD(net.parameters(),lr=.003, momentum=.9)

    if dp == True:
        print('adding privacy engine')
        # if we are training with differential privacy, create the engine
        privacy_engine = PrivacyEngine(
            net,
            batch_size=VIRTUAL_BATCH_SIZE,
            sample_size=len(trainset),
            alphas=range(2,32),
            noise_multiplier=NOISE_MULTIPLIER,
            max_grad_norm=MAX_GRAD_NORM
        )   
        privacy_engine.attach(optimizer)

    for epoch in range(3):  # currently training for 5 epochs
        print(f'epoch: {epoch}')
        train(net,trainloader,optimizer,epoch,device,dp)

def get_trained_model(dataset,dp):
    classes, trainloader, testloader, trainset, testset = get_test_train_loaders(dataset)
    device = torch.device("cuda")
    net = torchvision.models.alexnet(num_classes=len(classes)).to(device)

    if dp:
        net = module_modification.convert_batchnorm_modules(net)
    inspector = DPModelInspector()

    print(f"Model valid:        {inspector.validate(net)}")
    print(f"Model trained DP:   {dp}")
    net = net.to(device)
    
    if dp:
        PATH = './trained_models/' + dataset + '_dp' + '.pth'
    else:
        PATH = './trained_models/' + dataset + '.pth'
    # PATH='./trained_models/mnist_0_931.pth'
    # PATH='./trained_models/mnist_0_dp_316.pth'
    print(len(classes))
    net.load_state_dict(torch.load(PATH+"",map_location=torch.device('cpu')))
    return net,classes,trainloader,testloader,trainset,testset

def get_test_train_loaders(dataset):
    transform = transforms.Compose([
        transforms.Resize(64),
        # transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    print(f"using dataset:      {dataset}")
    if dataset == "mnist":
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine') 
    else: # ie == cifar
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return classes,trainloader, testloader, trainset, testset



def get_device(train):

    use_cuda = torch.cuda.is_available() and train
    # use_cuda = torch.cuda.is_available() 
    device = torch.device("cuda")
    return device