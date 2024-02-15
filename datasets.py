from torchvision import datasets,transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader,random_split
import tqdm



def Transformer():
    transformer=transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5,)),
                                    transforms.Resize((224,224))])
    return transformer




def loading_datasets(transformer,root):
    train_dataset=datasets.CIFAR10(root=root,  # LOAD CIFAR !!!!!
                                train=True,
                                download=True,
                                transform=transformer)
    
    test_dataset=datasets.CIFAR10(root=root,
                                  train=False,
                                  download=True,
                                  transform=transformer)

    
    return train_dataset,test_dataset




def random_split_fn(dataset,valid_size):
    
    valid_dataset,train_dataset=random_split(dataset,[valid_size,len(dataset)-valid_size])
    
    return valid_dataset,train_dataset



def Dataloader(train_dataset,test_dataset,valid_dataset,BATCH_SIZE):
    TrainDataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    TestDataloader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)
    ValidDataloader=DataLoader(valid_dataset,batch_size=BATCH_SIZE,shuffle=False)
    
    return TrainDataloader,TestDataloader,ValidDataloader



def Tensorboard_Image(dataloader,devices,tensorboard_writer,max_batch):
    print("Images Loading to Tensorboard")
    
    progress_bar=tqdm.tqdm(range(max_batch),desc="Tensorboard Process",position=0)
    
    for batch,(img,label) in enumerate(dataloader,0):
        img=img.to(devices)
        img_grid=make_grid(img,nrow=20)
        tensorboard_writer.add_image("Train_Image",img_grid)
        progress_bar.update(1)
        if batch==max_batch:
            break
    progress_bar.close()