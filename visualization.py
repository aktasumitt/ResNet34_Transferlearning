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

    
    

