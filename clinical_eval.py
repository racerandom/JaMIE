import copy
from collections import defaultdict
from data_utils import MultiheadConll


def calculate_f1(tps, fps, fns):
    p = 0. if not (tps + fps) else (tps / (tps + fps))
    r = 0. if not (tps + fns) else (tps / (tps + fns))
    f1 = 0. if not (p + r) else (2 * p * r / (p + r))
    return p, r, f1


def evaluate_tuples(pred_tuples, gold_tuples, ix2rel, rel_col=-1):
    # eval_dic[rel] = [tps, fps, fns]
    tps_id = 0
    fps_id = 1
    fns_id = 2
    eval_dic = defaultdict(lambda: [0, 0, 0])
    for g_t in gold_tuples:
        g_rel = ix2rel[g_t[rel_col]] if isinstance(g_t[rel_col], int) else g_t[rel_col]
        if g_rel in ['N', 'O']:
            continue
        if g_t in pred_tuples:
            eval_dic[g_rel][tps_id] += 1
            pred_tuples.remove(g_t)
        else:
            eval_dic[g_rel][fns_id] += 1
    for p_t in pred_tuples:
        p_rel = ix2rel[p_t[rel_col]] if isinstance(p_t[rel_col], int) else p_t[rel_col]
        if p_rel in ['N', 'O']:
            continue
        eval_dic[p_rel][fps_id] += 1

    print()
    for rel, (rel_tps, rel_fps, rel_fns) in eval_dic.items():
        p, r, f1 = calculate_f1(rel_tps, rel_fps, rel_fns)
        print("\t{:>12}, p {:.6f}, r {:.6f}, f1 {:.6f}, (tps {:d}, fps {:d}, fns {:d})".format(
            rel,
            p, r, f1,
            rel_tps, rel_fps, rel_fns
        ))

    all_tps = sum([v[tps_id] for v in eval_dic.values()])
    all_fps = sum([v[fps_id] for v in eval_dic.values()])
    all_fns = sum([v[fns_id] for v in eval_dic.values()])
    all_p, all_r, all_f1 = calculate_f1(all_tps, all_fps, all_fns)
    print("overall, p %.6f, r %.6f, f1 %.6f, (tps %i, fps %i, fns %i)\n" % (
        all_p, all_r, all_f1,
        all_tps, all_fps, all_fns
    ))


class TupleEvaluator(object):
    # eval_dic[rel] = [tps, fps, fns]
    def __init__(self):
        self.tps_id = 0
        self.fps_id = 1
        self.fns_id = 2
        self.eval_dic = defaultdict(lambda: [1e-10, 1e-10, 1e-10])

    def reset(self):
        self.eval_dic = defaultdict(lambda: [1e-10, 1e-10, 1e-10])

    def update(self, gold_tuples, pred_tuples, rel_col=-1):
        gold_tuple_cp = copy.deepcopy(gold_tuples)
        pred_tuple_cp = copy.deepcopy(pred_tuples)
        for g_t in gold_tuple_cp:
            g_rel = g_t[rel_col]
            if g_rel in ['N', 'O', '_']:
                continue
            if g_t in pred_tuple_cp:
                self.eval_dic[g_rel][self.tps_id] += 1
                pred_tuple_cp.remove(g_t)
            else:
                self.eval_dic[g_rel][self.fns_id] += 1
        for p_t in pred_tuple_cp:
            p_rel = p_t[rel_col]
            if p_rel in ['N', 'O', '_']:
                continue
            self.eval_dic[p_rel][self.fps_id] += 1

    def print_results(self, message, f1_mode, print_level):

        class_scores = {}
        for rel, (rel_tps, rel_fps, rel_fns) in self.eval_dic.items():
            p, r, f1 = calculate_f1(rel_tps, rel_fps, rel_fns)
            class_scores[rel] = (p, r, f1)
            if print_level > 1:
                print(f"\t{rel:>12}, p {p * 100:2.4f}, r {r * 100:2.4f}, f1 {f1 * 100:2.4f},"
                      f" (tps {rel_tps:.0f}, fps {rel_fps:.0f}, fns {rel_fns:.0f})")

        if f1_mode == 'micro':
            all_tps = sum([v[self.tps_id] for v in self.eval_dic.values()])
            all_fps = sum([v[self.fps_id] for v in self.eval_dic.values()])
            all_fns = sum([v[self.fns_id] for v in self.eval_dic.values()])
            all_p, all_r, all_f1 = calculate_f1(all_tps, all_fps, all_fns)
        elif f1_mode == 'macro':
            all_p = sum([v[0] for k, v in class_scores.items()]) / len(class_scores)
            all_r = sum([v[1] for k, v in class_scores.items()]) / len(class_scores)
            all_f1 = sum([v[2] for k, v in class_scores.items()]) / len(class_scores)
        else:
            raise ValueError(f"Unknown f1_model: {f1_mode} ...")

        if print_level >= 1:
            print(f"{message}, {f1_mode}, overall, p {all_p * 100:2.4f}, r {all_r * 100:2.4f}, f1 {all_f1 * 100:2.4f}")

        return all_f1


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
        return self._ner_evaluator.print_results('ner', f1_mode=self.f1_mode, print_level=print_level)

    def eval_mod(self, print_level=1):
        for s_gold_mod, s_pred_mod in zip(self._gold_mhs._mod_entities, self._pred_mhs._mod_entities):
            self._mod_evaluator.update(s_gold_mod, s_pred_mod, rel_col=-1)
        return self._mod_evaluator.print_results('mod', f1_mode=self.f1_mode, print_level=print_level)

    def eval_rel(self, print_level=1):
        for s_gold_rel, s_pred_rel in zip(self._gold_mhs._rel_detailed_triplets, self._pred_mhs._rel_detailed_triplets):
            self._rel_evaluator.update(s_gold_rel, s_pred_rel, rel_col=-1)
        return self._rel_evaluator.print_results('rel', f1_mode=self.f1_mode, print_level=print_level)

