import torch
import tqdm

def Train(EPOCHS,TrainDataloader,TestDataloader,optimizer,loss_fn,devices,model_vgg,callbakcs_path,save_callbacks,Tensorboard_Writer):

    training_values_list=[]
    step=0
    
    for epoch in range(EPOCHS):
        
        train_correct=0.0
        train_total=0.0
        loss_train=0.0
        
        progress_bar=tqdm.tqdm(range(len(TrainDataloader)),desc="Training Process",position=0)

        
        for batch,(img,label) in enumerate(TrainDataloader,0):
            
            img=img.to(devices)
            label=label.to(devices)
            
            optimizer.zero_grad()
            out=model_vgg(img)
            _,predict_train=torch.max(out,1)
            loss=loss_fn(out,label)
            loss.backward()
            optimizer.step()
            
            loss_train+=loss.item()
            train_correct+=(predict_train==label).sum().item()
            train_total+=label.size(0)
            progress_bar.update(1)

        
            if batch==len(TrainDataloader)-1:
                    
                    valid_correct=0.0
                    valid_total=0.0
                    loss_valid=0.0  
                    
                    for batch_valid,(img,label) in enumerate(TestDataloader,0):
            
                        img_valid=img.to(devices)
                        label_valid=label.to(devices)
                        with torch.no_grad():
                            out_valid=model_vgg(img_valid)
                        
                        _,predict_valid=torch.max(out_valid,1)
                        loss_val=loss_fn(out_valid,label_valid)
                        
                        loss_valid+=loss_val.item()
                        valid_correct+=(predict_valid==label_valid).sum().item()
                        valid_total+=label_valid.size(0)
            
                    # Adding Training Values to Tensorboard
                    Tensorboard_Writer.add_scalar("Loss Train",(loss_train/(batch+1)),global_step=step)
                    Tensorboard_Writer.add_scalar("Accuracy Train",(100*train_correct/train_total),global_step=step)
                    Tensorboard_Writer.add_scalar("Loss Valid",(loss_val/(batch_valid+1)),global_step=step)
                    Tensorboard_Writer.add_scalar("Accuracy Valid",(100*valid_correct/valid_total),global_step=step)
                    step+=1
            
                    training_values_dict={  "Epoch":epoch+1,
                                            "Loss Train":(loss_train/(batch+1)),
                                            "Accuracy Train":(100*train_correct/train_total),
                                            "Loss Valid":(loss_valid/(batch_valid+1)),
                                            "Accuracy Valid":(100*valid_correct/valid_total)}
                    
                    progress_bar.set_postfix(training_values_dict)
                    
                    
                    training_values_list.append(training_values_dict)
        
        progress_bar.close()
        
        save_callbacks(epoch=epoch+1,
                       optimizer=optimizer,
                       model=model_vgg,
                       callback_path=callbakcs_path,
                       )
        
    return training_values_list
        
        
        
        
        