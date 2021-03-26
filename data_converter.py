from utils import *
from transformers import *
from argparse import ArgumentParser


def xml_to_conll(xml_dir, conll_dir, doc_level, is_raw, segmenter, tokenizer):
    train_scale = 1.0
    with_dct = True
    xml_list = [os.path.join(xml_dir, file) for file in sorted(os.listdir(xml_dir)) if file.endswith(".xml")]
    print(f"total files: {len(xml_list)}")
    if not os.path.exists(conll_dir):
        os.makedirs(conll_dir)
    if not is_raw:
        batch_convert_document_to_conll(
            xml_list,
            os.path.join(
                conll_dir,
                f"single.conll"
            ),
            sent_tag=True,
            contains_modality=True,
            with_dct=with_dct,
            is_raw=is_raw,
            morph_analyzer_name=segmenter,
            bert_tokenizer=tokenizer,
            is_document=doc_level
        )
    else:
        for dir_file in xml_list:
            file_name = dir_file.split('/')[-1].rsplit('.', 1)[0]
            single_convert_document_to_conll(
                dir_file,
                os.path.join(
                    conll_dir,
                    f"{file_name}.conll"
                ),
                sent_tag=True,
                contains_modality=True,
                with_dct=with_dct,
                is_raw=is_raw,
                morph_analyzer_name=segmenter,
                bert_tokenizer=bert_tokenizer,
                is_document=doc_level
            )


def cross_validation(in_dir, out_dir, doc_level, is_raw, segmenter, cv_num, tokenizer):
    train_scale = 1.0
    with_dct = True
    cv_data_split = doc_kfold(in_dir, train_scale=train_scale, cv=cv_num)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for cv_id, (train_files, dev_files, test_files) in enumerate(cv_data_split):
        print(len(train_files), len(dev_files), len(test_files))
        batch_convert_document_to_conll(
            train_files,
            os.path.join(
                out_dir,
                f"cv{cv_id}_train.conll"
            ),
            sent_tag=True,
            contains_modality=True,
            with_dct=with_dct,
            is_raw=is_raw,
            morph_analyzer_name=segmenter,
            bert_tokenizer=tokenizer,
            is_document=doc_level
        )
        batch_convert_document_to_conll(
            dev_files,
            os.path.join(
                out_dir,
                f"cv{cv_id}_dev.conll"
            ),
            sent_tag=True,
            contains_modality=True,
            with_dct=with_dct,
            is_raw=False,
            morph_analyzer_name=segmenter,
            bert_tokenizer=tokenizer,
            is_document=doc_level
        )
        batch_convert_document_to_conll(
            test_files,
            os.path.join(
                out_dir,
                f"cv{cv_id}_test.conll"
            ),
            sent_tag=True,
            contains_modality=True,
            with_dct=with_dct,
            is_raw=False,
            morph_analyzer_name=segmenter,
            bert_tokenizer=tokenizer,
            is_document=doc_level
        )


def conll_to_xml(conll_dir, xml_dir):
    conll_list = [os.path.join(conll_dir, file) for file in sorted(os.listdir(conll_dir)) if file.endswith(".conll")]
    print(f"total files: {len(conll_list)}")
    if not os.path.exists(xml_dir):
        os.makedirs(xml_dir)
    for dir_conll in conll_list:
        file_name = dir_conll.split('/')[-1].rsplit('.', 1)[0]
        xml_out = os.path.join(xml_dir, f"{file_name}.xml")
        doc_conll = data_objects.MultiheadConll(dir_conll)
        doc_conll.doc_to_xml(xml_out)


parser = ArgumentParser(description='Convert xml to conll for training')

parser.add_argument("--mode", dest="mode",
                    help="convert_mode, xml2conll or conll2xml", metavar="CONVERT_MODE")

parser.add_argument("--xml", dest="xml_dir",
                    help="input xml dir")

parser.add_argument("--conll", dest="conll_dir",
                    help="output conll dir")

parser.add_argument("--doc_level",
                    action='store_true',
                    help="document-level extraction or sentence-level extraction")

parser.add_argument("--is_raw",
                    action='store_true',
                    help="whether the input xml is raw or annotated")

parser.add_argument("--cv_num", default=0, type=int,
                    help="k-fold cross-validation, 0 presents not to split data")

parser.add_argument("--segmenter", default='mecab', type=str,
                    help="segmenter: mecab (w/ MeCab package) and jumanpp (w/ pyknp package) ")

parser.add_argument("--bert_dir", type=str,
                    help="BERT dir for initializing tokenizer")

args = parser.parse_args()

if args.mode in ['xml2conll']:
    bert_tokenizer = BertTokenizer.from_pretrained(
        args.bert_dir,
        do_lower_case=False,
        do_basic_tokenize=False,
        tokenize_chinese_chars=False
    )
    bert_tokenizer.add_tokens(['[JASP]'])

    if args.cv_num == 0:
        xml_to_conll(args.xml_dir, args.conll_dir, args.doc_level, args.is_raw, args.segmenter, bert_tokenizer)
    elif args.cv_num > 0:
        cross_validation(args.xml_dir, args.conll_dir, args.doc_level, args.is_raw, args.segmenter, args.cv_num, bert_tokenizer)
    else:
        raise Exception(f"Incorrect cv number {args.cv_num}...")
elif args.mode in ['conll2xml']:
    conll_to_xml(args.conll_dir, args.xml_dir)
else:
    raise Exception(f"Unknown converting mode {args.mode}...")

