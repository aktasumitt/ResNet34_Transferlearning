import torch
import matplotlib.pyplot as plt

def visualization_dataset(dataset):
    
    for i in range(16):
        
        plt.subplot(4,4,i+1)
        plt.imshow(torch.transpose(torch.transpose(dataset[i][0],0,2),0,1))
        plt.xlabel(f"{dataset[i][1]}")
        plt.xticks([])
        plt.yticks([])
    
    plt.show()
    
    

def Visualization_test_predict(test_dataloader,prediction_test):
    
    for batch_test,(img,label) in enumerate(test_dataloader,0):
        
        plt.subplot(batch_test/4,4,batch_test+1)
        plt.imshow(torch.transpose(torch.transpose(img[0],0,2),0,1))
        plt.xlabel(f"predict: {prediction_test[batch_test][0]},real: {label[0]} ")
    
    plt.show()
    
    

