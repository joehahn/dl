# dl

by Joe Hahn,<br />
jmh.datasciences@gmail.com,<br />
16 January 2018<br />
git branch=master


### Intro:

this will eventually become a simple demo of deep learning on aws with keras and 
gpus...in progress...


### Setup:


1 Launch a g2.2xl EC2 instance in AWS using these settings:

    EC2 > launch instance > Community AMIs
    search for 'Bitfusion Ubuntu TensorFlow' > g2.2xlarge
    set tag Name=dl
    security group settings:
        set SSH and TCP entries to have Source=My IP         #this permits ssh and jupyter
        add custom TCP rule, port=6006, Source=My IP         #this permits tensorboard
    create keypair with name=dl2
    Launch

this instance provides 26 ECUs, 8 vCPUs, 2.6 GHz, Intel Xeon E5-2670, 15 Gb memory, 
10 Gb SSD Storage, cost=$0.76/hr

2 store private ssh key tf-demo.pem in subfolder 'private' with these permissions:

    chmod 400 private/dl2.pem

3 obtain the instance's public IP address from the EC2 console, and then ssh into the instance:

    ssh -i private/dl2.pem ubuntu@ec2-54-218-50-120.us-west-2.compute.amazonaws.com

4 clone this repo:

    git clone https://github.com/joehahn/dl.git
    cd dl
    mkdir data

5 purchase 5 years of NYSE data from eoddata.com, and store that data in desktop's 'data' folder

6 use scp on desktop to upload data to aws instance

    scp -i private/dl2.pem data/*.zip  ubuntu@ec2-54-218-50-120.us-west-2.compute.amazonaws.com:/home/ubuntu/dl/data

7 unpack data on the EC2 instance:

    unzip 'data/NYSE*.zip' -d data
    
7 install additional python libraries

    sudo pip install seaborn

8 update locate database:

    sudo updatedb

9 get instance-id:

    ec2metadata --instance-id

10 change this line in ~/.jupyter/jupyter_notebook_config.py, so Jupyter stores its notebooks in dl:

    c.NotebookApp.notebook_dir = u'/home/ubuntu/dl'

11 start jupyter:

    jupyter notebook

12 browse jupyter at public_IP:8888 and log in with password=instance-id

    ec2-54-218-50-120.us-west-2.compute.amazonaws.com:8888





8 Train a CNN on CIFAR-10 images:

    cd ~/tensorflow-models/tutorials/image/cifar10
    python ./cifar10_multi_gpu_train.py

9 start tensorboard:

    cd ~/tensorflow-models/tutorials/image/cifar10
    tensorboard --logdir .

then browse

    ec2-54-190-198-117.us-west-2.compute.amazonaws.com:6006

and log in with password=instance-id

10 Monitor GPU usage:

    watch -n0.1 nvidia-smi



