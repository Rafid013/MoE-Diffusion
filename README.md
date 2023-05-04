# Dataset Download
The CIFAR-10 64x64 dataset has to be downloaded from this [link](https://www.kaggle.com/datasets/joaopauloschuler/cifar10-64x64-resized-via-cai-super-resolution). This cannot be downloaded without logging in to the website. That's why we are not adding a download script for this dataset. After downloading the dataset, it has to be unzipped to the main project folder. After unzipping, there should be two resultant folders at the following locations


*MoE-Diffusion/cifar10-32/*

*MoE-Diffusion/cifar10-64/*


To get the file to ARC, we downloaded the dataset to our PC and then used the SCP command to upload it to ARC.

# Required Installation
The following installations are required.
```
pip install pytorch-fid
conda install -c conda-forge pytorch-lightning
```

# Running the codes
The following commands will run the full experiment
```
python train_model.py
python testing.py
chmod 777 fid_metric_1.sh
chmod 777 fid_metric_2.sh
./fid_metric_1.sh
./fid_metric_2.sh
python binary_accuracy.py
```

- train_model.py file trains the model
- testing.py file generates images using the trained model
- fid_metric_1.sh script file computes FID scores for images generated using 1 label
- fid_metric_2.sh script file computes FID scores for images generated using 2 labels
- binary_accuracy.py file computes the binary classification accuracy using a pre-trained ResNet18 model

# Special Thanks
- Our implementation of diffusion model is taken from this GitHub [repo](https://github.com/dome272/Diffusion-Models-pytorch)
- To compute the FID score, we use the library found in this [link](https://github.com/mseitzer/pytorch-fid)
- To compute the accuracy, we used the pre-trained ResNet18 found in this GitHub [repo](https://github.com/huyvnphan/PyTorch_CIFAR10)
