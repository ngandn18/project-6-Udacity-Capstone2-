# The project 6 in course Udacity Machine Learning Nano Degree

## Capstone Project 2 - Image classification model - AWS SageMaker

This project implements Pytorch image classification model on AWS Sagemaker.

All the main ideas are witten in file ***proposal.pdf*** and ***report.pdf***.

Because the tuning and training time too long, the connections from Aws studio get lost, then I seperate the process into three notebook to follow easily and clearly:

1. data_process.ipynb

2. tuning_training.ipynb

3. deploy_test.ipynb

## All in my submission

### 1. Folder code

- ***data_process.ipynb***: transform from cifar.tgz into data image files.
- ***tuning_training.ipynb***: perform tuning and training model on aws sagemaker.
- ***hpo.py***: entry point for tuning job.
- ***train_model.py***: entry point for training job.
- ***deploy_test.ipynb***: deploy the endpoint and test all the test image on this endpoint.
- ***inference.py***: entry point for deploe the endpoint.

### 2. Folder csv - csv.zip

- All the csv files contain related data and info are used for this project.
- This folder is zipped into one file csv.zip for fast upload.

### 3. Folder images

- All the images are saved when my notebook executed on Aws Sagemaker Studio, tuning job, training jobs, cloud watch and endpoints.

### 4. File ***proposal.pdf***

### 5. File ***report.pdf***

### 6. Files ***data_process.html***, ***tuning_training.html***, and ***deploy_test.html*** for easy review

[Here is my blogpost](https://ngandn18.github.io/project/proj6.html)

### Thank you for all your helpful and valuable supports
