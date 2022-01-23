import json
import logging
import sys
import os
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
def Net():
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


def model_fn(model_dir):
    print("In model_fn. Model directory is -")
    print(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    
    with open(os.path.join(model_dir, "model.pt"), "rb") as f:
        print("Loading the cifar-10 model")
        checkpoint = torch.load(f , map_location =device)
        model.load_state_dict(checkpoint)
        print('MODEL-LOADED')
        logger.info('model loaded successfully')
    model.eval()
    return model


def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    # process an image uploaded to the endpoint
    logger.debug(f'Request body CONTENT-TYPE is: {content_type}')
    logger.debug(f'Request body TYPE is: {type(request_body)}')
    if content_type == JPEG_CONTENT_TYPE: 
        return Image.open(io.BytesIO(request_body))
    logger.debug('SO loded JPEG content')
    # process a URL submitted to the endpoint
    
    if content_type == JSON_CONTENT_TYPE:
        #img_request = requests.get(url)
        logger.debug(f'Request body is: {request_body}')
        request = json.loads(request_body)
        logger.debug(f'Loaded JSON object: {request}')
        url = request['url']
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content))
    
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# inference
def predict_fn(input_object, model):
    logger.info('In predict fn')
    img_size = 224
    test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    logger.info("transforming input")
    input_object=test_transform(input_object)
    # resnet model doesn't need this adjustment
    # One new line for this densenet model - predictor wants torch [1,3,...] 
    input_object = input_object[:3,:,:]
    input_object = input_object.to(device)
    with torch.no_grad():
        logger.info("Calling model")
        prediction = model(input_object.unsqueeze(0))
    return prediction

