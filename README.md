## JaMIE: a Japanese Medical Information Extraction toolkit


## pipeline processes: 

* [Preprocess] of converting raw text to CONLL-styple input data
* [Medical Entity Recognition (MER)] [link to the trained MER model] (https://drive.google.com/)
* [Modality Classification (RC)] [link to the trained MC model] (https://drive.google.com/)
* [Relation Extraction (RE)] [link to the trained RE model] (https://drive.google.com/)
* [Postprocess] of converting the CONLL-style output to the XML file

## step1: [Preprocess] of converting raw text to CONLL-styple input data:

Convert XML files to CONLL files for Train/Test. You can also convert raw text to CONLL-style for Test.

> python data_converter.py \\
>    --mode xml2conll \\
>    --xml $XML_FILES_DIR \\
>    --conll $OUTPUT_CONLL_DIR \\
>    --cv_num 0 \ # n-fold cross-validation, 0 for single output 
>    --segmenter mecab \ # please use mecab and NICT bert
>    --bert_dir $PRETRAINED_BERT \ # BERT tokenizer dir 
>    --is_raw  # whether the input is raw text    

## step2: [Medical Entity Recognition]

### Train:

> python clinical_pipeline_ner.py \\
> --pretrained_model $PRETRAINED_BERT_DIR \\
> --saved_model $DIR_TO_SAVE_MODEL \\ 
> --train_file $TRAIN_CONLL_FILE \\
> --dev_file $DEV_CONLL_FILE \\
> --batch_size 16 \\
> --do_train 

### Test:

> python clinical_pipeline_ner.py \\  
> --saved_model $SAVED_MODEL_DIR \\ 
> --test_file $TEST_CONLL_IN \\
> --test_output $TEST_CONLL_OUTPUT \\
> --batch_size  


## step3: [Modality Classification]

### Train:

> python clinical_pipeline_mod.py \\
> --pretrained_model $PRETRAINED_BERT_DIR \\
> --saved_model $DIR_TO_SAVE_MODEL \\ 
> --train_file $TRAIN_CONLL_FILE \\
> --dev_file $DEV_CONLL_FILE \\
> --batch_size 16 \\
> --do_train 

### Test:

> python clinical_pipeline_mod.py \\  
> --saved_model $SAVED_MODEL_DIR \\ 
> --test_file $TEST_CONLL_IN \\
> --test_output $TEST_CONLL_OUTPUT \\
> --batch_size  

## step4: [Relation Extraction]

### Train:

> python clinical_pipeline_rel.py \\
> --pretrained_model $PRETRAINED_BERT_DIR \\
> --saved_model $DIR_TO_SAVE_MODEL \\ 
> --train_file $TRAIN_CONLL_FILE \\
> --dev_file $DEV_CONLL_FILE \\
> --batch_size 16 \\
> --do_train 

### Test:

> python clinical_pipeline_rel.py \\  
> --saved_model $SAVED_MODEL_DIR \\ 
> --test_file $TEST_CONLL_IN \\
> --test_output $TEST_CONLL_OUTPUT \\
> --batch_size

### step5: [Postprocess] of converting the CONLL-style output to the XML file
> python data_converter.py \ \
>    --mode conll2xml \ \
>    --xml $XML_FILES_DIR \ \
>    --conll $OUTPUT_CONLL_DIR 

## Required Package
pytorch=>1.3.1\
transformers=2.2.2\
mojimoji\
tqdm\
python-textformatting\
gensim\
scikit-learn\
pandas\
apex(for fp16)\
Mophological analyzer required:\
pyknp\
MeCab\
*jumanpp\
*mecab (juman-dict)\
Pretrained BERT required:\
*NICT-BERT (NICT_BERT-base_JapaneseWikipedia_32K_BPE)





