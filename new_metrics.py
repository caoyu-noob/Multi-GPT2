# author: Xiang Gao @ Microsoft Research, Oct 2018
# compute NLP evaluation metrics

import io
import re
import subprocess
import sys
import time
import itertools
from collections import defaultdict, Counter

import numpy as np

py_version = sys.version.split('.')[0]
if py_version == '2':
    open = io.open
else:
    unicode = str

def str2bool(s):
    # to avoid issue like this: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if s.lower() in ['t', 'true', '1', 'y']:
        return True
    elif s.lower() in ['f', 'false', '0', 'n']:
        return False
    else:
        raise ValueError


def calc_nist(path_refs, path_hyp, fld_out='temp', n_lines=None):
    return calc_nist_bleu(path_refs, path_hyp, fld_out, n_lines)[0]


def calc_bleu(path_refs, path_hyp, fld_out='temp', n_lines=None):
    return calc_nist_bleu(path_refs, path_hyp, fld_out, n_lines)[1]


def calc_nist_bleu(path_refs, path_hyp, fld_out='temp', n_lines=None):
    # call mteval-v14c.pl
    # ftp://jaguar.ncsl.nist.gov/mt/resources/mteval-v14c.pl
    # you may need to cpan install XML:Twig Sort:Naturally String:Util

    if n_lines is None:
        n_lines = len(open(path_hyp, encoding='utf-8').readlines())
    if fld_out is None:
        fld_out = 'temp'
    _write_xml([''], fld_out + '/src.xml', 'src', n_lines=n_lines)
    _write_xml([path_hyp], fld_out + '/hyp.xml', 'hyp', n_lines=n_lines)
    _write_xml(path_refs, fld_out + '/ref.xml', 'ref', n_lines=n_lines)

    time.sleep(1)
    cmd = [
        'perl','metrics/mteval-v14c.pl',
        '-s', '%s/src.xml'%fld_out,
        '-t', '%s/hyp.xml'%fld_out,
        '-r', '%s/ref.xml'%fld_out,
        ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output, error = process.communicate()

    lines = output.decode().split('\n')
    try:
        nist_score = lines[-22].strip('\r').split()[3]
        bleu_score = lines[-22].strip('\r').split()[7]
        nist = lines[-6].strip('\r').split()[1:5]
        bleu = lines[-4].strip('\r').split()[1:5]
        return float(nist_score), float(bleu_score), [float(x) for x in nist], [float(x) for x in bleu]

    except Exception:
        print('mteval-v14c.pl returns unexpected message')
        print('cmd = '+str(cmd))
        print(output.decode())
        print(error.decode())
        return [-1]*4, [-1]*4


def calc_cum_bleu(path_refs, path_hyp):
    # call multi-bleu.pl
    # https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl
    # the 4-gram cum BLEU returned by this one should be very close to calc_nist_bleu
    # however multi-bleu.pl doesn't return cum BLEU of lower rank, so in nlp_metrics we preferr calc_nist_bleu
    # NOTE: this func doesn't support n_lines argument and output is not parsed yet

    # process = subprocess.Popen(
    #         ['perl', 'metrics/multi-bleu.perl'] + path_refs,
    #         stdout=subprocess.PIPE,
    #         stdin=subprocess.PIPE
    #         )
    process = subprocess.Popen(
        ['perl', 'metrics/multi-bleu.perl'] + path_refs,
        stdout=subprocess.PIPE,
        stdin=open(path_hyp, encoding='utf-8')
    )
    # with open(path_hyp, encoding='utf-8') as f:
    #     lines = f.readlines()
    # for i,line in enumerate(lines):
    #     process.stdin.write(line.encode())
    #     print(i)
    output, error = process.communicate()
    return output.decode()


def calc_meteor(path_refs, path_hyp, fld_out='temp', n_lines=None, pretokenized=True):
    # Call METEOR code.
    # http://www.cs.cmu.edu/~alavie/METEOR/index.html

    path_merged_refs = fld_out + '/refs_merged.txt'
    _write_merged_refs(path_refs, path_merged_refs)

    cmd = [
            'java', '-Xmx1g',	# heapsize of 1G to avoid OutOfMemoryError
            '-jar', 'metrics/meteor-1.5/meteor-1.5.jar',
            path_hyp, path_merged_refs, 
            '-r', str(len(path_refs)), 	# refCount
            '-l', 'en', '-norm' 	# also supports language: cz de es fr ar
            ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    for line in output.decode().split('\n'):
        if "Final score:" in line:
            return float(line.split()[-1])

    print('meteor-1.5.jar returns unexpected message')
    print("cmd = " + " ".join(cmd))
    print(output.decode())
    print(error.decode())
    return -1 


def calc_entropy(path_hyp, n_lines=None):
    # based on Yizhe Zhang's code
    etp_score = [0.0,0.0,0.0,0.0]
    counter = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]
    i = 0
    for line in open(path_hyp, encoding='utf-8'):
        i += 1
        words = line.strip('\n').split()
        for n in range(4):
            for idx in range(len(words)-n):
                ngram = ' '.join(words[idx:idx+n+1])
                counter[n][ngram] += 1
        if i == n_lines:
            break

    for n in range(4):
        total = sum(counter[n].values())
        for v in counter[n].values():
            etp_score[n] += - v /total * (np.log(v) - np.log(total))

    return etp_score


def calc_avg_len(path, n_lines=None):
    l = []
    for line in open(path, encoding='utf8'):
        l.append(len(line.strip('\n').split()))
        if len(l) == n_lines:
            break
    return np.mean(l)


def calc_div(path_hyp):
    tokens = [0.0, 0.0]
    types = [defaultdict(int), defaultdict(int)]
    for line in open(path_hyp, encoding='utf-8'):
        words = line.strip('\n').split()
        for n in range(2):
            for idx in range(len(words)-n):
                ngram = ' '.join(words[idx:idx+n+1])
                types[n][ngram] = 1
                tokens[n] += 1
    div1 = len(types[0].keys())/tokens[0] if tokens[0] != 0 else 0
    div2 = len(types[1].keys())/tokens[1] if tokens[1] != 0 else 0
    return [div1, div2]


def nlp_metrics(path_refs, path_hyp, fld_out='temp',  n_lines=None):
    nist, bleu = calc_nist_bleu(path_refs, path_hyp, fld_out, n_lines)
    meteor = calc_meteor(path_refs, path_hyp, fld_out, n_lines)
    entropy = calc_entropy(path_hyp, n_lines)
    div = calc_div(path_hyp)
    avg_len = calc_avg_len(path_hyp, n_lines)

    return nist, bleu, meteor, entropy, div, avg_len


def specified_nlp_metric(path_refs, path_hyp, metric):
    i = None

    m = re.search('_[\d]\Z', metric)
    if m:
        metric, i = metric[:m.span()[0]], int(metric[m.span()[0]+1:]) - 1

    try:
        res = eval(f'calc_{metric}(path_refs, path_hyp)')
    except:
        res = eval(f'calc_{metric}(path_hyp)')

    return res if i is None else res[i]


def _write_merged_refs(paths_in, path_out, n_lines=None):
    # prepare merged ref file for meteor-1.5.jar (calc_meteor)
    # lines[i][j] is the ref from i-th ref set for the j-th query

    lines = []
    for path_in in paths_in:
        lines.append([line.strip('\n') for line in open(path_in, encoding='utf-8')])

    with open(path_out, 'w', encoding='utf-8') as f:
        for j in range(len(lines[0])):
            for i in range(len(paths_in)):
                f.write(unicode(lines[i][j]) + "\n")


def _write_xml(paths_in, path_out, role, n_lines=None):
    # prepare .xml files for mteval-v14c.pl (calc_nist_bleu)
    # role = 'src', 'hyp' or 'ref'

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE mteval SYSTEM "">',
        '<!-- generated by https://github.com/golsun/NLP-tools -->',
        '<!-- from: %s -->'%paths_in,
        '<!-- as inputs for ftp://jaguar.ncsl.nist.gov/mt/resources/mteval-v14c.pl -->',
        '<mteval>',
        ]

    for i_in, path_in in enumerate(paths_in):

        # header ----

        if role == 'src':
            lines.append('<srcset setid="unnamed" srclang="src">')
            set_ending = '</srcset>'
        elif role == 'hyp':
            lines.append('<tstset setid="unnamed" srclang="src" trglang="tgt" sysid="unnamed">')
            set_ending = '</tstset>'
        elif role == 'ref':
            lines.append('<refset setid="unnamed" srclang="src" trglang="tgt" refid="ref%i">'%i_in)
            set_ending = '</refset>'
        
        lines.append('<doc docid="unnamed" genre="unnamed">')

        # body -----

        if role == 'src':
            body = [''] * n_lines
        else:
            with open(path_in, 'r', encoding='utf-8') as f:
                body = f.readlines()
            if n_lines is not None:
                body = body[:n_lines]
        for i in range(len(body)):
            line = body[i].strip('\n')
            line = line.replace('&',' ').replace('<',' ')		# remove illegal xml char
            if len(line) == 0:
                line = '__empty__'
            lines.append('<p><seg id="%i"> %s </seg></p>'%(i + 1, line))

        # ending -----

        lines.append('</doc>')
        if role == 'src':
            lines.append('</srcset>')
        elif role == 'hyp':
            lines.append('</tstset>')
        elif role == 'ref':
            lines.append('</refset>')

    lines.append('</mteval>')
    with open(path_out, 'w', encoding='utf-8') as f:
        f.write(unicode('\n'.join(lines)))

def normalize_answer(s):
    re_art = re.compile(r'\b(a|an|the)\b')
    re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    # return white_space_fix(remove_articles(remove_punc(lower(s))))
    return white_space_fix(remove_articles(lower(s)))

def _f1_score(ref, pred):
    """
    Compute precision, recall and f1 given a set of gold and prediction items.

    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values

    :return: tuple (p, r, f1) for precision, recall, f1
    """
    # ref_items = normalize_answer(ref).split()
    # pred_items = normalize_answer(pred).split()
    ref_items = ref.split()
    pred_items = pred.split()
    common = Counter(ref_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(ref_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

def get_f1_score(refs_list, preds_list):
    f1 = 0
    for i in range(len(refs_list)):
        f1 += _f1_score(refs_list[i], preds_list[i])[2]
    return f1 / len(refs_list)

def _split_into_words(sentences):
  """Splits multiple sentences into words and flattens the result"""
  return list(itertools.chain(*[_.split(" ") for _ in sentences]))

def _get_ngrams(n, text):
  """Calcualtes n-grams.
  Args:
    n: which n-grams to calculate
    text: An array of tokens
  Returns:
    A set of n-grams
  """
  ngram_set = set()
  text_length = len(text)
  max_index_ngram_start = text_length - n
  for i in range(max_index_ngram_start + 1):
    ngram_set.add(tuple(text[i:i + n]))
  return ngram_set

def _get_word_ngrams(n, sentences):
  """Calculates word n-grams for multiple sentences.
  """
  assert len(sentences) > 0
  assert n > 0

  words = _split_into_words(sentences)
  return _get_ngrams(n, words)

def rouge_n(evaluated_sentences, reference_sentences, n=2):
  """
  Computes ROUGE-N of two text collections of sentences.
  Sourece: http://research.microsoft.com/en-us/um/people/cyl/download/
  papers/rouge-working-note-v1.3.1.pdf
  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentences: The sentences from the referene set
    n: Size of ngram.  Defaults to 2.
  Returns:
    A tuple (f1, precision, recall) for ROUGE-N
  Raises:
    ValueError: raises exception if a param has len <= 0
  """
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")

  evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
  reference_ngrams = _get_word_ngrams(n, reference_sentences)
  reference_count = len(reference_ngrams)
  evaluated_count = len(evaluated_ngrams)

  # Gets the overlapping ngrams between evaluated and reference
  overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
  overlapping_count = len(overlapping_ngrams)

  # Handle edge case. This isn't mathematically correct, but it's good enough
  if evaluated_count == 0:
    precision = 0.0
  else:
    precision = overlapping_count / evaluated_count

  if reference_count == 0:
    recall = 0.0
  else:
    recall = overlapping_count / reference_count

  f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

  # return overlapping_count / reference_count
  return f1_score, precision, recall

def _f_p_r_lcs(llcs, m, n):
  """
  Computes the LCS-based F-measure score
  Source: http://research.microsoft.com/en-us/um/people/cyl/download/papers/
  rouge-working-note-v1.3.1.pdf
  Args:
    llcs: Length of LCS
    m: number of words in reference summary
    n: number of words in candidate summary
  Returns:
    Float. LCS-based F-measure score
  """
  r_lcs = llcs / m
  p_lcs = llcs / n
  beta = p_lcs / (r_lcs + 1e-12)
  num = (1 + (beta**2)) * r_lcs * p_lcs
  denom = r_lcs + ((beta**2) * p_lcs)
  f_lcs = num / (denom + 1e-12)
  return f_lcs, p_lcs, r_lcs

def _len_lcs(x, y):
  """
  Returns the length of the Longest Common Subsequence between sequences x
  and y.
  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
  Args:
    x: sequence of words
    y: sequence of words
  Returns
    integer: Length of LCS between x and y
  """
  table = _lcs(x, y)
  n, m = len(x), len(y)
  return table[n, m]

def _lcs(x, y):
  """
  Computes the length of the longest common subsequence (lcs) between two
  strings. The implementation below uses a DP programming algorithm and runs
  in O(nm) time where n = len(x) and m = len(y).
  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
  Args:
    x: collection of words
    y: collection of words
  Returns:
    Table of dictionary of coord and len lcs
  """
  n, m = len(x), len(y)
  table = dict()
  for i in range(n + 1):
    for j in range(m + 1):
      if i == 0 or j == 0:
        table[i, j] = 0
      elif x[i - 1] == y[j - 1]:
        table[i, j] = table[i - 1, j - 1] + 1
      else:
        table[i, j] = max(table[i - 1, j], table[i, j - 1])
  return table

def rouge_l_sentence_level(evaluated_sentences, reference_sentences):
  """
  Computes ROUGE-L (sentence level) of two text collections of sentences.
  http://research.microsoft.com/en-us/um/people/cyl/download/papers/
  rouge-working-note-v1.3.1.pdf
  Calculated according to:
  R_lcs = LCS(X,Y)/m
  P_lcs = LCS(X,Y)/n
  F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)
  where:
  X = reference summary
  Y = Candidate summary
  m = length of reference summary
  n = length of candidate summary
  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentences: The sentences from the referene set
  Returns:
    A float: F_lcs
  Raises:
    ValueError: raises exception if a param has len <= 0
  """
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")
  reference_words = _split_into_words(reference_sentences)
  evaluated_words = _split_into_words(evaluated_sentences)
  m = len(reference_words)
  n = len(evaluated_words)
  lcs = _len_lcs(evaluated_words, reference_words)
  return _f_p_r_lcs(lcs, m, n)

def get_rouge(refs_list, preds_list):
  """Calculates average rouge scores for a list of hypotheses and
  references"""

  rouge_1 = [
      rouge_n([pred], [ref], 1) for pred, ref in zip(preds_list, refs_list)
  ]
  rouge_1_f, rouge_1_p, rouge_1_r = map(np.mean, zip(*rouge_1))

  # Calculate ROUGE-2 F1, precision, recall scores
  rouge_2 = [
      rouge_n([pred], [ref], 2) for pred, ref in zip(preds_list, refs_list)
  ]
  rouge_2_f, rouge_2_p, rouge_2_r = map(np.mean, zip(*rouge_2))

  # Calculate ROUGE-L F1, precision, recall scores
  rouge_l = [
      rouge_l_sentence_level([hyp], [ref])
      for hyp, ref in zip(preds_list, refs_list)
  ]
  rouge_l_f, rouge_l_p, rouge_l_r = map(np.mean, zip(*rouge_l))
  return np.mean([rouge_1_f,rouge_2_f,rouge_l_f]), rouge_1_f, rouge_2_f, rouge_l_f

def cal_dist(preds_list):
    ngram1, ngram2 = set(), set()
    sentence_dist1, sentence_dist2 = 0, 0
    total_length = 0
    for pred in preds_list:
        length = len(pred.split(' '))
        cur_gram_1 = _get_word_ngrams(1, [pred])
        cur_gram_2 = _get_word_ngrams(2, [pred])
        if length > 0:
            sentence_dist1 += len(cur_gram_1) / length
        if length > 1:
            sentence_dist2 += len(cur_gram_2) / (length - 1)
        ngram1 = ngram1.union(cur_gram_1)
        ngram2 = ngram2.union(cur_gram_2)
        total_length += length
    if total_length == len(preds_list):
        return sentence_dist1 / len(preds_list), sentence_dist2 / len(preds_list), 0, 0
    return sentence_dist1 / len(preds_list), sentence_dist2 / len(preds_list), len(ngram1) / total_length, \
           len(ngram2) / (total_length - len(preds_list))

def calc_entropy(preds, n_lines=None):
    # based on Yizhe Zhang's code
    entropy_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]
    i = 0
    for line in preds:
        i += 1
        words = line.strip('\n').split()
        for n in range(4):
            for idx in range(len(words)-n):
                ngram = ' '.join(words[idx:idx+n+1])
                counter[n][ngram] += 1
        if i == n_lines:
            break

    for n in range(4):
        total = sum(counter[n].values())
        for v in counter[n].values():
            entropy_score[n] += - v /total * (np.log(v) - np.log(total))

    return entropy_score

def nlp_metrics(ref_file, pred_file, root_path=None):
    preds_list = []
    with open(pred_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            preds_list.append(normalize_answer(line))
    refs_list = []
    with open(ref_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            refs_list.append(normalize_answer(line))

    sentence_dist1, sentence_dist2, corpus_dist1, corpus_dist2 = cal_dist(preds_list)
    s_dist= [sentence_dist1, sentence_dist2]
    c_dist = [corpus_dist1, corpus_dist2]
    entropy_scores = calc_entropy(preds_list)

    nist, nist_bleu, nist_list, nist_bleu_list = calc_nist_bleu([ref_file], pred_file, fld_out=root_path)

    # meteor_score = calc_meteor([ref_file], pred_file)

    bleu_output = calc_cum_bleu([ref_file], pred_file)
    bleu_text = re.search(r"BLEU = (.+?), (.+?)/(.+?)/(.+?)/(.+?) \(", bleu_output)

    bleu = float(bleu_text.group(1))
    bleu1 = float(bleu_text.group(2))
    bleu2 = float(bleu_text.group(3))
    bleu3 = float(bleu_text.group(4))
    bleu4 = float(bleu_text.group(5))
    bleu_list = [bleu1, bleu2, bleu3, bleu4]

    f1_score = get_f1_score(refs_list, preds_list)
    _, _, _, rouge_l = get_rouge(refs_list, preds_list)
    avg_pred_length = np.mean([len(x.split(' ')) for x in preds_list])

    return bleu, bleu_list, nist, nist_list, nist_bleu, nist_bleu_list, s_dist, c_dist, entropy_scores, 0.0, \
           rouge_l, f1_score, avg_pred_length
