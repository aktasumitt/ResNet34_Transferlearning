import torch
import datasets
import model
import testing
import training
import visualization
import config
import callbacks
import predict
from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.filterwarnings("ignore")


# Create SummaryWriter for Training Items to Use On Tensorborad
Tensorboard_Writer = SummaryWriter(log_dir=config.TENSORBOARD_LOG_DIR)


# Check Cuda
devices = ("cuda" if torch.cuda.is_available() else "cpu")


# Transformer Datasets
transformer = datasets.Transformer()


# Create Cifar100 Dataset
train_dataset,test_dataset = datasets.loading_datasets(transformer=transformer, root=config.LOAD_DATASET_ROOT)

# Random split for validation data
valid_dataset, train_dataset = datasets.random_split_fn(train_dataset,valid_size=config.VALID_SIZE)


# Visualize image (but we dont need because of the Tensorboard)
visualization.visualization_dataset(dataset=train_dataset)


# Create Dataloaders with Batchsize
TrainDataloader, TestDataloader, ValidDataloader = datasets.Dataloader(train_dataset=train_dataset,
                                                                       test_dataset=test_dataset,
                                                                       valid_dataset=valid_dataset,
                                                                       BATCH_SIZE=config.BATCH_SIZE)

# Adding Train Images on The Tensorboard
datasets.Tensorboard_Image(dataloader=TrainDataloader,
                           devices=devices,
                           tensorboard_writer=Tensorboard_Writer,
                           max_batch=config.MAX_BATCH_TENSORBOARD_IMG)


# Create Model VGG19 and freezing parameters.
VGG = model.Loading_Resnet().to(devices)


# We will added Last layer for our dataset
Model = model.Add_Last_Layer(model=VGG, label_size=10)

Model.to(devices)


# Create Optimizer and Loss Function
optimizer = torch.optim.Adam(params=Model.parameters(), lr=config.LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()


# Loading Callbacks
if config.LOAD_CALLBACK == True:

    callback = torch.load(config.PATH_CALLBACK)
    starting_epoch = callbacks.load_callbacks(callback=callback,
                                              optimizer=optimizer,
                                              Model=Model)

else:
    starting_epoch = 0
    print("Training is starting from scratch...")


# Training
if config.TRAIN==True:
    training_values_list = training.Train(EPOCHS=config.EPOCHS,
                                          TrainDataloader=TrainDataloader,
                                          TestDataloader=ValidDataloader,
                                          optimizer=optimizer,
                                          loss_fn=loss_fn,
                                          devices=devices,
                                          model_vgg=Model,
                                          callbakcs_path=config.PATH_CALLBACK,
                                          save_callbacks=callbacks.save_callbacks,
                                          Tensorboard_Writer=Tensorboard_Writer)


# Testing
if config.TEST == True:
    results_test, prediction_test_list = testing.TEST_MODEL(TestDataloader, model_vgg=Model, loss_fn=loss_fn, devices=devices)


# Prediction and Visualization
if config.PREDICTION==True:
    predict.predict(prediction_folder_path=config.PREDICTION_PATH,
                    model=Model,
                    idx_class_dict=test_dataset.class_to_idx,
                    visualize_img=True)
