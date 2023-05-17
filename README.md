## JaMIE: a Japanese Medical Information Extraction toolkit

[comment]: <> (## [PRISM] Medical Tag recognition and Disease certainty classification)

[comment]: <> (## pipeline processes: )

[comment]: <> (### [medical tag recognition] -> [disease certainty classification])

[comment]: <> (## Install)

[comment]: <> (> git clone URL  )

[comment]: <> (> cd XX)

[comment]: <> (Copy the processed data &#40;in 黒橋研 server&#41; into the 'data' folder in XX. )

[comment]: <> (## step1: medical tag recognition:)

[comment]: <> (### Train and test:)

[comment]: <> (> python clinical\_ner.py \\  )

[comment]: <> (> --corpus 'goku' \\  )

[comment]: <> (> --model 'checkpoints/ner/' \\ # save model   )

[comment]: <> (> --epoch 5 \\  )

[comment]: <> (> --batch 16 \\  )

[comment]: <> (> --do_train )

[comment]: <> (### Test:)

[comment]: <> (> python clinical\_ner.py \\  )

[comment]: <> (> --corpus 'goku' \\  )

[comment]: <> (> --model 'checkpoints/ner/' # load model  )

[comment]: <> (Predicted texts will be located in the 'outputs' folder.)

[comment]: <> (### Evaluation:)

[comment]: <> (> cd conlleval  )

[comment]: <> (> python conlleval.py < ../outputs/ner\_goku\_ep5\_eval.txt)

[comment]: <> (## step2: disease certainty classification)

[comment]: <> (### Train and test:)

[comment]: <> (> python clinical\_cert.py \\  )

[comment]: <> (> --corpus 'goku' \\  )

[comment]: <> (> --model 'checkpoints/cert/' \\ # save model  )

[comment]: <> (> --ner\_out 'outputs/ner\_goku\_ep3\_out.txt' \\  # predicted ner results with BIO format  )

[comment]: <> (> --epoch 3 \\  )

[comment]: <> (> --batch 16 \\  )

[comment]: <> (> --do_train )

[comment]: <> (### Test:)

[comment]: <> (> python clinical\_cert.py \\  )

[comment]: <> (> --corpus 'goku' \\  )

[comment]: <> (> --model 'checkpoints/cert/' # load model  )

[comment]: <> (> --ner\_out 'outputs/ner\_goku\_ep3\_out.txt'   # predicted ner results with BIO format)

[comment]: <> (Predicted texts will be located in the 'outputs' folder.)

## Joint Japanese Medical Problem, Modality and Relation Recognition

The Train/Test phrases require all train, dev, test file converted to CONLL-style. Please check data_converter.py

### Installation (python3.8)
> git clone https://github.com/racerandom/JaMIE.git \
> cd JaMIE \
#### Required python package
> pip install -r requirements.txt 

#### Mophological analyzer required:
[mecab (juman-dict)](https://taku910.github.io/mecab/) by default \
[jumanpp](https://github.com/ku-nlp/jumanpp) 

#### Pretrained BERT required for training:
[NICT-BERT (NICT_BERT-base_JapaneseWikipedia_32K_BPE)](https://alaginrc.nict.go.jp/nict-bert/index.html)


## Train: 
### Pre-processing: Batch Converter from XML (or raw text) to CONLL for Train/Test

Convert XML or raw text files to CONLL files before Train/Test. You can also convert raw text to CONLL-style for Test, but please make sure the file extension is .xml.

> python data_converter.py \ \
>    --mode xml2conll \ \
>    --xml $XML_FILES_DIR \ \
>    --conll $OUTPUT_CONLL_DIR \ \
>    --cv_num 5 \ # 5-fold cross-validation, 0 presents to generate single conll file\
>    --doc_level \ # generate document-level ([SEP] denotes sentence boundaries) or sentence-level conll files\
>    --segmenter mecab \ # please use mecab and NICT bert currently\
>    --bert_dir $PRETRAINED_BERT 

### Start training：  
> CUDA_VISIBLE_DEVICES=$SEED python clinical_joint.py \ \
>    --pretrained_model $PRETRAINED_BERT \ \
>    --train_file $TRAIN_FILE \ \
>    --dev_file $DEV_FILE \ \
>    --dev_output $DEV_OUT \ \
>    --saved_model $MODEL_DIR_TO_SAVE \ \
>    --enc_lr 2e-5 \ \
>    --batch_size 4 \ \
>    --warmup_epoch 2 \ \
>    --num_epoch 20 \ \
>    --do_train \ \
>    --fp16 (apex required)

The models trained on radiography interpretation reports of Lung Cancer (LC) and general medical reports of Idiopathic Pulmonary Fibrosis (IPF) are to be availabel: 
* [The trained model of radiography interpretation reports of Lung Cancer](https://drive.google.com/file/d/1Xh-XA8rusO-fKr6z1gaiyYUNqnBODNaq/view?usp=sharing)
* [The trained model of case reports of Idiopathic Pulmonary Fibrosis](https://drive.google.com/file/d/1hrKdz4mW5Wp9lwM_ZuTbO0UjoMfu-Dy3/view?usp=sharing)

## Test:
> CUDA_VISIBLE_DEVICES=$SEED python clinical_joint.py \ \
>    --saved_model $SAVED_MODEL \ \
>    --test_file $TEST_FILE \ \
>    --test_output $TEST_OUT \ \
>    --batch_size 4

### Batch Converter from predicted CONLL to XML
> python data_converter.py \ \
>    --mode conll2xml \ \
>    --xml $XML_FILES_DIR \ \
>    --conll $OUTPUT_CONLL_DIR 


## Citation
If you use our code in your research, please cite the following papers:
```bibtex
@inproceedings{cheng-etal-2022-jamie,
   title={JaMIE: A Pipeline Japanese Medical Information Extraction System with Novel Relation Annotation},
      author={Fei Cheng, Shuntaro Yada, Ribeka Tanaka, Eiji Aramaki, Sadao Kurohashi},
         booktitle={Proceedings of the Thirteenth Language Resources and Evaluation Conference (LREC 2022)},
            year={2022}
}
@inproceedings{cheng2021jamie,
   title={JaMIE: A Pipeline Japanese Medical Information Extraction System},
      author={Fei Cheng, Shuntaro Yada, Ribeka Tanaka, Eiji Aramaki, Sadao Kurohashi},
         booktitle={arXiv},
            year={2021}
}
@inproceedings{yada-etal-2020-towards,
   title={Towards a Versatile Medical-Annotation Guideline Feasible Without Heavy Medical Knowledge: Starting From Critical Lung Diseases},
      author={Shuntaro Yada, Ayami Joh, Ribeka Tanaka, Fei Cheng, Eiji Aramaki, Sadao Kurohashi},
         booktitle={Proceedings of the Twelfth Language Resources and Evaluation Conference (LREC 2020)},
            year={2020}
}
@inproceedings{cheng-etal-2020-dynamically,
   title={Dynamically Updating Event Representations for Temporal Relation Classification with Multi-category Learning},
      author={Fei Cheng, Masayuki Asahara, Ichiro Kobayashi, Sadao Kurohashi},
         booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020), Findings Volume},
            year={2020}
}
```
