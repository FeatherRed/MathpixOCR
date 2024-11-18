import nltk
import distance
import torch.cuda
from nltk.translate.bleu_score import sentence_bleu
from ptflops import get_model_complexity_info

# references = ["I love cats", "The sun is shining", "Hello, world!"]
# hypotheses = ["I love dogs", "The moon is bright", "Hello, everyone!"]

# bilingual evaluation understudy score:
def bleu_sore(references, hypotheses):
    bleu_score = 0.0
    for i, j in zip(references, hypotheses):
        bleu_score += max(sentence_bleu([i], j), 0.01)
    bleu_score = bleu_score / len(references) * 100
    return bleu_score


# edit distance
def edit_distance(references, hypotheses):
    '''Computes Levenshtein distance between two sequences.
    Args:
    references: list of sentences (one hypothesis)
    hypotheses: list of sentences (one hypothesis)
    Returns:
    1 - levenshtein distance: (higher is better, 1 is perfect)
    '''
    d_leven, len_tot = 0, 0
    for ref, hypo in zip(references, hypotheses):
        d_leven += distance.levenshtein(ref, hypo)
        len_tot += float(max(len(ref), len(hypo)))
    return (1. - d_leven / len_tot) * 100


# exact math score:
def exact_match_score(references, hypothesis):
    assert len(references) == len(hypothesis), 'references and hypothesis have different length!!!'
    count = 0
    for ref, hypo in zip(references, hypothesis):
        if ref == hypo:
            count += 1
    Accuracy = count / len(references)
    Exact_Match_Score = Accuracy * 100
    return Exact_Match_Score

def model_complexity(model, x_shape):
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model.cuda(),
            x_shape,
            as_strings = True,
            print_per_layer_stat = False
        )
        print('{:<30} {:<8}'.format('Computational complexity: ', macs))
        print('{:<30} {:<8}'.format('Number of parameters: ', params))

def model_complexity_score(new_model, old_model, x_shape):
    with torch.cuda.device(0):
        new_macs, new_params = get_model_complexity_info(
            new_model.cuda(),
            x_shape,
            as_strings = False,
            print_per_layer_stat = False
        )

        old_macs, old_params = get_model_complexity_info(
            old_model.cuda(),
            x_shape,
            as_strings = False,
            print_per_layer_stat = False
        )

    Model_Complexity_Score = max(0, min(100, (new_macs - old_macs) / (0.1 * new_macs)))
    return Model_Complexity_Score

def total_score(references, hypotheses):
    # 去除空格
    ref = [r.strip().replace(' ', '') for r in references]
    hyp = [h.strip().replace(' ', '') for h in hypotheses]
    print(f'debug: ref:{ref[0]}, {len(ref)}, hyp:{hyp[0]}, {len(ref)}')
    blue_s = bleu_sore(ref, hyp)
    edit_d = edit_distance(ref, hyp)
    exact_s = exact_match_score(ref, hyp)

    return blue_s, edit_d, exact_s, (blue_s + edit_d + exact_s) / 3

def overall_score(references, hypotheses, x_shape):
    # todo: 这里应该传入两个模型对象，一个新的 一个旧的，才能计算复杂度
    ref = [r.strip().replace(' ', '') for r in references]
    hyp = [h.strip().replace(' ', '') for h in hypotheses]
    # print(f'debug: ref:{ref[0]}, {len(ref)}, hyp:{hyp[0]}, {len(ref)}')
    blue_s = bleu_sore(ref, hyp)
    edit_d = edit_distance(ref, hyp)
    exact_s = exact_match_score(ref, hyp)
    model_s = model_complexity_score()

    return blue_s, edit_d, exact_s, model_s, (blue_s + edit_d + exact_s + model_s) / 4