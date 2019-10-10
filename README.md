## [PRISM] Medical Tag recognition and Disease certainty classification

## pipeline processes: 

### [medical tag recognition] -> [disease certainty classification]

## Install
> git clone URL  
> cd XX

Copy the processed data (in 黒橋研 server) into the 'data' folder in XX. 

## step1: medical tag recognition:

### Train and test:
> python clinical\_ner.py \\  
> --corpus 'goku' \\  
> --model 'checkpoints/ner/' \\ # save model   
> --epoch 5 \\  
> --batch 16 \\  
> --do_train 

### Test:
> python clinical\_ner.py \\  
> --corpus 'goku' \\  
> --model 'checkpoints/ner/' # load model  

Predicted texts will be located in the 'outputs' folder.

### Evaluation:
> cd conlleval  
> python conlleval.py < ../outputs/ner\_goku\_ep5\_eval.txt

## step2: disease certainty classification

### Train and test:
> python clinical\_cert.py \\  
> --corpus 'goku' \\  
> --model 'checkpoints/cert/' \\ # save model  
> --ner\_out 'outputs/ner\_goku\_ep3\_out.txt' \\  # predicted ner results with BIO format  
> --epoch 3 \\  
> --batch 16 \\  
> --do_train 

### Test:
> python clinical\_cert.py \\  
> --corpus 'goku' \\  
> --model 'checkpoints/cert/' # load model  
> --ner\_out 'outputs/ner\_goku\_ep3\_out.txt'   # predicted ner results with BIO format

Predicted texts will be located in the 'outputs' folder.


## Required Package
pytorch=1.0.0  
pytorch-pretrained-bert=0.6.2  
mojimoji  
tqdm  
pyknp  
jumanpp 






