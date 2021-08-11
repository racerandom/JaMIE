## JaMIE: a Japanese Medical Information Extraction toolkit
A pipeline Japanese Medical IE system, which offers automatic and accurate analysis of medical entities, modalities and relations for radiography interpretation reports of Lung Cancer (LC) and general medical reports of Idiopathic Pulmonary Fibrosis (IPF).

## Installation

### Clone from Github

```bash
$git clone -b demo https://github.com/racerandom/JaMIE.git
cd JaMIE
```

### Required Environment
```bash
Python packages required:
pytorch=>1.3.1
transformers=2.2.2
mojimoji
tqdm
python-textformatting
gensim
scikit-learn
pandas
apex #for fp16
Mophological analyzer required:
pyknp
MeCab
*jumanpp
*mecab (juman-dict)
Pretrained BERT required:
*NICT-BERT (NICT_BERT-base_JapaneseWikipedia_32K_BPE)
```

## Pipeline Usage: 

* [Preprocess] of converting raw text to CONLL-styple input data
* [Medical Entity Recognition (MER)] 
* [Modality Classification (RC)] 
* [Relation Extraction (RE)] 
* [Postprocess] of converting the CONLL-style output to the XML file

### Step1: [Preprocess] of converting raw text to CONLL-styple input data:

Convert XML files to CONLL files for Train/Test. You can also convert raw text to CONLL-style for Test.

```bash
python data_converter.py \
--mode xml2conll \
--xml $XML_FILES_DIR \
--conll $OUTPUT_CONLL_DIR \
--cv_num 0 \ # n-fold cross-validation, 0 is single output
--segmenter mecab \ # please use mecab and NICT bert
--bert_dir $PRETRAINED_BERT \ # BERT tokenizer dir
--is_raw  # whether the input is raw text    
```
### Step2: [Medical Entity Recognition (MER)]

#### Train:
```bash
python clinical_pipeline_ner.py \
--pretrained_model $PRETRAINED_BERT_DIR \
--saved_model $DIR_TO_SAVE_MODEL \
--train_file $TRAIN_CONLL_FILE \
--dev_file $DEV_CONLL_FILE \
--batch_size 16 \
--do_train 
```

* We are still in the application of the permission for openning the trained model. Once it's accepted, the following links here will be replaced by the real.

<a href="drive.google.com" target="_top">Trained LC MER model<a>\
<a href="drive.google.com" target="_top">Trained IPF MER mode<a>

#### Test:
```bash
python clinical_pipeline_ner.py \
--saved_model $SAVED_MODEL_DIR \
--test_file $TEST_CONLL_IN \
--test_output $TEST_CONLL_OUTPUT \
--batch_size  
```

### Step3: [Modality Classification (MC)]

#### Train:
```bash
python clinical_pipeline_mod.py \
--pretrained_model $PRETRAINED_BERT_DIR \
--saved_model $DIR_TO_SAVE_MODEL \
--train_file $TRAIN_CONLL_FILE \
--dev_file $DEV_CONLL_FILE \
--batch_size 16 \
--do_train 
```
* We are still in the application of the permission for openning the trained model. Once it's accepted, the following links here will be replaced by the real.

<a href="drive.google.com" target="_top">Trained LC MC model<a>\
<a href="drive.google.com" target="_top">Trained IPF MC mode<a>

#### Test:
```bash
python clinical_pipeline_mod.py \
--saved_model $SAVED_MODEL_DIR \
--test_file $TEST_CONLL_IN \
--test_output $TEST_CONLL_OUTPUT \
--batch_size  
```
### Step4: [Relation Extraction (RE)]

#### Train:
```bash
python clinical_pipeline_rel.py \
--pretrained_model $PRETRAINED_BERT_DIR \
--saved_model $DIR_TO_SAVE_MODEL \
--train_file $TRAIN_CONLL_FILE \
--dev_file $DEV_CONLL_FILE \
--batch_size 16 \
--do_train 
```
* We are still in the application of the permission for openning the trained model. Once it's accepted, the following links here will be replaced by the real.

<a href="drive.google.com" target="_top">Trained LC RE model<a>\
<a href="drive.google.com" target="_top">Trained IPF RE mode<a>

#### Test:
```bash
python clinical_pipeline_rel.py \
--saved_model $SAVED_MODEL_DIR \
--test_file $TEST_CONLL_IN \
--test_output $TEST_CONLL_OUTPUT \
--batch_size
```
### Step5: [Postprocess] of converting the CONLL-style output to the XML file
```bash
python data_converter.py \
--mode conll2xml \
--xml $XML_FILES_DIR \
--conll $OUTPUT_CONLL_DIR 
```






