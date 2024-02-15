import torch,tqdm

def TEST_MODEL(test_dataloader,model_vgg,loss_fn,devices):
    
    prediction_list=[]
    test_correct=0.0
    test_total=0.0
    loss_test_values=0.0  
    prog_bar=tqdm.tqdm(range(len(test_dataloader),"Test Progress"))              
    for batch_test,(img,label) in enumerate(test_dataloader,0):
            
        
        img_test=img.to(devices)
        label_test=label.to(devices)
        
        with torch.no_grad():
            out_test=model_vgg(img_test)
        
        _,predict_test=torch.max(out_test,1)
        loss_test=loss_fn(out_test,label_test)
        
                        
        loss_test_values+=loss_test.item()
        test_correct+=(predict_test==label_test).sum().item()
        test_total+=label_test.size(0)
        
        prediction_list.append(predict_test)
        
        prog_bar.update(1)
        
    
    accuracy_test=100*test_correct*test_total
    prog_bar.set_postfix({"Loss":loss_test_values/(batch_test+1),"Accuracy":accuracy_test})
    
    return {"Loss":loss_test_values/(batch_test+1),"Accuracy":accuracy_test},prediction_list