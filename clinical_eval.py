from data_utils import MultiheadConll
from utils import TupleEvaluator


class MhsEvaluator(object):
    def __init__(self, gold_mhs_file, pred_mhs_file, f1_mode='micro'):
        self._gold_mhs = MultiheadConll(gold_mhs_file)
        self._pred_mhs = MultiheadConll(pred_mhs_file)
        self.f1_mode = f1_mode
        self._ner_evaluator = TupleEvaluator()
        self._mod_evaluator = TupleEvaluator()
        self._rel_evaluator = TupleEvaluator()

    def eval_ner(self, print_level=1):
        for s_gold_ner, s_pred_ner in zip(self._gold_mhs._entities, self._pred_mhs._entities):
            self._ner_evaluator.update(s_gold_ner, s_pred_ner, rel_col=0)
        self._ner_evaluator.print_results('ner', f1_mode=self.f1_mode, print_level=1)

    def eval_mod(self, print_level=1):
        for s_gold_mod, s_pred_mod in zip(self._gold_mhs._mod_entities, self._pred_mhs._mod_entities):
            self._mod_evaluator.update(s_gold_mod, s_pred_mod, rel_col=-1)
        self._mod_evaluator.print_results('mod', f1_mode=self.f1_mode, print_level=1)

    def eval_rel(self, print_level=1):
        for s_gold_rel, s_pred_rel in zip(self._gold_mhs._rel_triplets, self._pred_mhs._rel_triplets):
            self._rel_evaluator.update(s_gold_rel, s_pred_rel, rel_col=-1)
        self._rel_evaluator.print_results('rel', f1_mode=self.f1_mode, print_level=1)