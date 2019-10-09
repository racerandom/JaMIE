## PRISM Medical Tag recognition and Disease certainty classification

## pipeline processes: 

### [medical tag recognition] -> [disease certainty classification]

## Install
> git clone URL  
> cd XX

Copy the processed data (in 黒橋研 server) into the 'data' folder in XX. 

## step1: medical tag recognition:

### Train a model:
> python clinical\_ner.py \\  
> --corpus 'goku' \\  
> --model 'checkpoints/ner/' \\ # save model   
> --epoch 3 \\  
> --batch 16 \\  
> --do_train 

### Test:
> python clinical\_ner.py \\  
> --corpus 'goku' \\  
> --model 'checkpoints/ner/' # load model  

Predicted texts will be located in the 'outputs' folder.

## step2: disease certainty classification

### Train a model:
> python clinical\_ner.py \\  
> --corpus 'goku' \\  
> --model 'checkpoints/ner/' \\ # save model   
> --epoch 3 \\  
> --batch 16 \\  
> --do_train 

### Test:
> python clinical\_ner.py \\  
> --corpus 'goku' \\  
> --model 'checkpoints/ner/' # load model  

Predicted texts will be located in the 'outputs' folder.




