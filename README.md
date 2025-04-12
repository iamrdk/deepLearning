# deepLearning
A repo to store simplified deep learning programs.

About 'deep_supervised_learning_images":
This file works on the simple concept of:
1) You have a dataset of images
2) You want to create a model that can classify them

All possible within 8GB of VRAM.

I have tried to find a lot of code bases that are working and also accurate. I managed to found a lot of code bases which are using MNIST (greyscale images), so when you feed RGB images, it won't work or you need to change a lot of things. This code can handle RGBs and it works on a variety of datasets, I have tested with celeb_face, animals, cars, places, etc. and all seems to be working fine within 8GB of VRAM (the code uses 6.9 GB).

I have tested with resnet18, resnet50 and convnext_tiny, and found that with my datasets, resnet50 performed the best. It took less time and was accurate too.

I have been training a lot of models like that, I have seen many epochs where the accuracy and loss peaks at one epoch and later drops due to overfitting or underfitting, thus, this stores a best value for the accuracy and when the accuracy keeps dropping, it makes sure the best model is saved and terminates the training, the max number of epochs is et to 100, but in my usage, it never went past 15 and always ended before.

There is also a plot shown after every model generation regarding the training specs and saves it. It also tries to predict few of the image samples and saves them as well.

Few requirements:
1) Have a NVIDIA GPU with CUDA support. If you have less than 8GB VRAM, try reducing the batch size to 32 and it should fit properly.
2) The structure of your dataset should be like:
    dataset/
    ├── class1/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── class2/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    └── ...
    If your dataset has train and test folders, just merge the contents inside the class. We validate using the entire class images by doing a validation split of 0.2
3) Make sure the name you pass to Image_Trainer is unique as it will create the models in the workspace itself.

Enjoy.
