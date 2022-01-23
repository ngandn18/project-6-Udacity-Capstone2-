# Import your dependencies.
# For instance, below are some dependencies you might need 
# if you are using Pytorch

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from torch.optim import lr_scheduler

import argparse
import os
import time
import copy
import sys
import io

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'


import smdebug.pytorch as smd
from smdebug.core.modes import ModeKeys

from smdebug import modes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
def net():
    '''
    TODO: Complete this function that initializes our model
          Remember to use a pretrained model
    '''
    num_classes = 10
    modelname='densenet161'
    model = models.densenet161(pretrained=True)
    # reset final fully connected layer
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
                            nn.Linear(num_features, 256),  
                            nn.ReLU(), 
                            nn.Dropout(0.3),
                            nn.Linear(256, num_classes))
    model = model.to(device)
    print(modelname)
    return model


def test(model, test_loader, criterion, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    hook.set_mode(modes.EVAL)
    model.eval()   
    
    running_loss=0
    running_corrects=0
    dt_sizes = 0
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # optimizer.zero_grad()

        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        dt_sizes += inputs.size(0)

    test_loss = running_loss / dt_sizes
    test_acc = running_corrects.double() / dt_sizes
    
    print(f'Test Loss: {test_loss}, Test Accu: {test_acc}')
   
    logger.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}\n".format(
            test_loss, test_acc)
        )


def train(model, dataloaders, criterion, optimizer, scheduler, dataset_sizes, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''

    begin = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    num_epochs = 15

    # create custom hook that has a customized forward function, 
    # so that we can get gradients of outputs
#     hook = CustomHook.create_from_json_file()
#     hook.register_module(model)

    # Training loop
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # set hook training phase
        hook.set_mode(modes.TRAIN)
        model.train()

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                hook.set_mode(modes.TRAIN)
                model.train()  # Set model to training mode
            else:
                hook.set_mode(modes.EVAL)
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device).requires_grad_()
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    train_time = time.time() - begin
    print('Training complete in {:.0f}m {:.0f}s'.format(
        train_time // 60, train_time % 60))
    print('Best valid Accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model    


def create_data_loaders(data_dir, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''

    # Updat mean, std 2022-01-13 cifar-10
    mean = [0.437687221685991, 0.4735875158811764, 0.4905213840275673]
    std = [0.2475770728439971, 0.2313360013602289, 0.2351076898744994]

    torch.manual_seed(18)
    
    img_size = 224

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    data_transforms['valid'] = data_transforms['test']

    image_datasets = {x: ImageFolder(
        os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'test', 'valid']}
  
    loaders = {x: DataLoader(image_datasets[x], 
        batch_size=batch_size,
        shuffle=True, num_workers=0) 
        for x in ['train', 'test', 'valid']}
 
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test', 'valid']}
    class_names = image_datasets['train'].classes

    return loaders, dataset_sizes, class_names

def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    '''
    TODO: Creat data
    '''   
    loaders, dataset_sizes,  class_names = create_data_loaders(args.data_dir, args.batch_size)
    # train_loader, test_loader, valid_loader = loaders.values()

    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    # Register the SMDebug hook to save output tensors.
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0.0005, weight_decay=0.0005) 
    #using SGD
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    hook.register_loss(loss_criterion)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)    
    model=train(model, loaders, loss_criterion, optimizer, scheduler, dataset_sizes, hook)
    '''
    TODO: Test the model to see its accuracy
    '''
    # test(model, test_loader, loss_criterion)
    test(model, loaders['test'], loss_criterion, hook)    
    '''
    TODO: Save the trained model
    '''
    # with open(os.path.join(args.model_dir, 'model.pt'), 'wb') as f:
    #     torch.save(model.state_dict(), f)
    # Very import to load for reusing model.state_dict().
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pt"))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument(
        "--batch_size", type=int, default=16, metavar="N",
        help="batch_size for training (default: 16)",
    )
    
    parser.add_argument(
        "--lr", type=float, default=0.1, metavar="LR", 
        help="learning rate (default: 1.0)"
    )
   
    parser.add_argument('--data_dir', type=str, 
                        default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model_dir', type=str, 
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, 
                        default=os.environ['SM_OUTPUT_DATA_DIR'])
   
    args=parser.parse_args()
    
    main(args)
