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

## Joint Japanese Medical Problem, Modality and Relation Recognition

### Train：  
> CUDA_VISIBLE_DEVICES=#SEED python clinical_joint.py \
>    --pretrained_model #PRETRAINED_MODEL_DIR \
>    --train_file #TRAIN_FILE \
>    --dev_file #DEV_FILE \
>    --dev_output #DEV_OUT \
>    --saved_model #MODEL_DIR_TO_SAVE \
>    --enc_lr 2e-5 \
>    --batch_size 4 \
>    --warmup_epoch 2 \
>    --num_epoch 20 \
>    --do_train \

### Test:
> CUDA_VISIBLE_DEVICES=#SEED python clinical_joint.py \
>    --saved_model #SAVED_MODEL \
>    --test_file #TEST_FILE \
>    --test_output #TEST_OUT \
>    --batch_size 4


### Convert XML to CONLL for training
> python data_converter.py \
> --xml #XML_FILES_DIR \
> --conll #OUTPUT_CONLL_DIR \
> --cv_num 5 \ # 5-fold cross-validation, 0 presents to generate single conll file 
> --doc_level \ # generate document-level ([SEP] denotes sentence boundaries) or sentence-level conll files
> --segmenter mecab \ # please use mecab and NICT bert currently
> --bert_dir ~/Tools/NICT_BERT-base_JapaneseWikipedia_32K_BPE

## Required Package
pytorch=1.3.1  
transformers=2.2.2
mojimoji  
tqdm  
pyknp 
MeCab
*jumanpp 
*mecab






