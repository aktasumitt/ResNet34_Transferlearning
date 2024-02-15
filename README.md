# ResNet34 Transfer Learning 

## Introduction:

- In this project, I aimed to finetune a pretrained ResNet34 model with Cifar10 Dataset.
- We can quickly train our dataset using the transfer learning method and obtain results.
-  The models we will use here have been trained for a long time on large datasets, so they usually perform well on our small datasets by using the weights from there. This way, we can reach results much faster. 
-  However, since the models here are not trained entirely with our data, they may produce worse results compared to training from scratch, especially with large datasets, but it is certain that they will be faster.

## Dataset:
- I used the Cifar10 dataset for this project, which consists of 10 labels with total 60000 train and 10000 test images.
- I randomly split the train dataset into training and validation with validation size 10000.
- And I divided them into mini-batches with a batch size of 100. 
- For the ResNet34 model, I resized images to (224x224x3) and applied normalization.
- Link For More information and downloading dataset: https://www.cs.toronto.edu/~kriz/cifar.html

## Train:
- I freezed the all layer of Model and i setted just last full connected layer according to my dataset (10 label)
- So I didnt train all model, i trained just last fc layer. 
- I used CrossEntropy Loss and Adam optimizer with 0.01 learning rate

## Results:
- After 6 epochs, the model achieved approximately 85% accuracy on both the training, validation, and test sets..

## Usage: 
- You can train the model by setting "TRAIN" to "True" in config file and your checkpoint will save in "config.CALLBACKS_PATH"
- Then you can predict the images placed in the Prediction folder by setting the "Load_Model" and "Prediction" values to "True" in the config file.







