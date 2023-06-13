from itertools import combinations, product
import itertools # TODO wtf
from json import load
import os
#from multiprocessing.sharedctypes import Value
import random
import pickle
from typing import List, Dict
import collections
import math
import pkg_resources

import numpy as np
import pandas as pd

from tokenizers import Tokenizer, SentencePieceBPETokenizer
from tokenizers import models, normalizers, pre_tokenizers, decoders, trainers
import morfessor
from morfessor.baseline import NumMorphCorpusWeight, MorphLengthCorpusWeight

def make_full_fname(fname):
    try:
        return pkg_resources.resource_filename('segmenters', 'data/' + fname)
    except ModuleNotFoundError:
        return 'data/' + fname

def load_celex_morpho():
    fpath = make_full_fname('celex_morpho_segmentation_curated.tsv')
    celex_morpho = pd.read_csv(fpath, sep='\t', header=None, index_col=0)
    celex_morpho = celex_morpho[1].sort_index()
    celex_morpho = celex_morpho.iloc[:-1]
    #celex_morpho.index = celex_morpho.index.map(lambda x: x.replace("'", ''))
    celex_morpho = celex_morpho.groupby(celex_morpho.index).agg(lambda x: '|'.join(set(x)))
    celex_morpho = celex_morpho[~celex_morpho.index.str.contains('-')]
    celex_morpho['.'] = '.'
    celex_morpho['?'] = '?'
    celex_morpho['!'] = '!'
    celex_morpho['nt'] = 'nt'
    celex_morpho["n't"] = 'nt'
    celex_morpho['is'] = 'is'
    # add no apostrophe 
    for key in celex_morpho.index[celex_morpho.index.str.contains("'")]:
        key_no_apo = key.replace("'", "")
        if key_no_apo not in celex_morpho:
            celex_morpho[key_no_apo] = celex_morpho[key]
    celex_morpho = celex_morpho.map(lambda x: x.split('-'))
    return celex_morpho

def list_built_in():
    names = [fname.split('.')[0] for fname in os.listdir(make_full_fname('')) if fname != 'celex_morpho_segmentation_curated.tsv']
    return ['celex_morpho', 'morfessor', 'bpe'] + names

def built_in(name):
    if name == 'celex_morpho':
        #fpath = pkg_resources.resource_filename('kgp', 'grammars/' + source + '.xml')
        celex_morpho = load_celex_morpho()
        return LookUpSegmenter(celex_morpho)
    
    if name == 'morfessor':
        return SegmenterMorfessor.load(make_full_fname('morfessor_tokens.pkl'))

    if name == 'bpe':
        return SegmenterBPE.load(make_full_fname('BPE_19251.json'))

    try:
        if 'morfessor' in name:
            fname = make_full_fname(name + '.pkl')
            return SegmenterMorfessor.load(fname)
        if 'sentencepiece' in name: 
            fname = make_full_fname(name + '.pkl')
            return SegmenterSentencePiece.load(fname)
        if 'wordpiece' in name: 
            fname = make_full_fname(name + '.pkl')
            return SegmenterWordPiece.load(fname)
        if 'unigram' in name: 
            fname = make_full_fname(name + '.bin')
            return SegmenterUnigram.load(fname)
        if 'BPE' in name: 
            fname = make_full_fname(name + '.json')
            return SegmenterBPE.load(fname)
    except FileNotFoundError:
        pass

    raise ValueError("invalid segmenter name")

class SegmentationError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class Segmenter:
    def __init__(self, sents: List[str] = []):
        pass
    def segment(self, sent: str):
        pass
    def save(self, fpath):
        with open(fpath, 'wb') as file:
            pickle.dump(self, file)
    def load(fpath):
        with open(fpath, 'rb') as file:
            segmenter = pickle.load(file)
        return segmenter

class SegmenterNotSerializable(Segmenter):
    def __init__(self):
        super().__init__([])
    def save(self, fpath):
        raise Exception("cant serialize this segmenter")
    def load(self, fpath):
        raise Exception("cant serialize this segmenter")

class CharacterSegmenter(SegmenterNotSerializable):
    def segment(self, sent: str):
        return [chr for chr in sent]

class ViterbiSegmenter(SegmenterNotSerializable):
    def __init__(self, score_fn):
        self.score_fn = score_fn
    def segment(self, sent: str):
        return viterbi_segment(sent, score_fn=self.score_fn)

class SegmenterDummy(SegmenterNotSerializable):
    def segment(self, sent: str):
        return [sent]

class SegmenterBlaBla(SegmenterNotSerializable):
    def segment(self, sent: str):
        return [sent[i*3:(i+1)*3] for i in range(int(math.ceil(len(sent)/3)))]

class FallBackSegmenter(SegmenterNotSerializable):
    def __init__(self, *segmenters):
        self.segmenters = segmenters
    def segment(self, sent: str):
        for segmenter in self.segmenters: 
            try: 
                return segmenter.segment(sent)
            except SegmentationError: 
                pass
        return [sent]

class LookUpSegmenter(Segmenter):
    def __init__(self, token2segments):
        self.token2segments = token2segments
    def segment(self, sent: str):
        try: 
            return self.token2segments[sent]
        except KeyError: 
            raise SegmentationError("segmentation rule not found")

class SegmenterMorfessor(Segmenter):
    def __init__(self, sents, num_morph_types=None, morph_length=None, corpusweight=1.0, threshold=0.01):
        self.model=None
        if len(sents) > 0:
            if type(sents[0]) == str: 
                sents = [sent.split() for sent in sents]
            tokens_flat = itertools.chain.from_iterable(sents)
            counter = collections.Counter(tokens_flat)
            train_data = [(count, token) for token, count in counter.items()]
            self.model = morfessor.BaselineModel(corpusweight=corpusweight)

            updater = None
            if num_morph_types is not None: 
                updater = NumMorphCorpusWeight(num_morph_types, threshold)
            elif morph_length is not None: 
                updater = MorphLengthCorpusWeight(morph_length, threshold)
            if updater is not None:
                self.model.set_corpus_weight_updater(updater)
                
            if len(train_data) > 0:
                self.model.load_data(train_data)
                self.model.train_batch()

    def segment(self, sent: str):
        if self.model == None:
            raise "must train me first"
        seg, score = self.model.viterbi_segment(sent)
        return seg

    def load(fpath):
        io = morfessor.MorfessorIO()
        model = io.read_binary_model_file(fpath)
        segmenter = SegmenterMorfessor([])
        segmenter.model = model
        return segmenter

    def save(self, fpath):
        io = morfessor.MorfessorIO()
        io.write_binary_model_file(fpath, self.model)

class SegmenterBPE(Segmenter):
    def __init__(self, sents, vocab_size=1000, show_progress=False):
        self.model = Tokenizer(models.BPE())
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            continuing_subword_prefix='',
            end_of_word_suffix='',
            show_progress=show_progress,
            special_tokens=['[UNK]']
        )
        self.model.train_from_iterator(sents, trainer=trainer)

    def segment(self, sent: str):
        return self.model.encode(sent).tokens

    def load(fpath):
        model = Tokenizer.from_file(fpath)
        segmenter = SegmenterBPE([])
        segmenter.model = model
        return segmenter

    def save(self, fpath):
        self.model.save(fpath)

class SegmenterUnigram(Segmenter):
    def __init__(self, sents, vocab_size=1000, max_piece_length=6, show_progress=False):
        self.model = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            max_piece_length=max_piece_length,
            show_progress=show_progress,
            unk_token='[UNK]'
        )
        self.model.train_from_iterator(sents, trainer=trainer)
    def segment(self, sent: str):
        return self.model.encode(sent).tokens
    def load(fpath):
        model = Tokenizer.from_file(fpath)
        segmenter = SegmenterUnigram([])
        segmenter.model = model
        return segmenter
    def save(self, fpath):
        self.model.save(fpath)

class SegmenterSentencePiece(Segmenter):
    def __init__(self, sents, vocab_size=1000, show_progress=False):
        self.model = SentencePieceBPETokenizer()
        self.model.train_from_iterator(
            sents, 
            vocab_size=vocab_size,
            show_progress=show_progress)
    def segment(self, sent: str):
        stupid_char = '▁'
        tokens_raw = self.model.encode(sent).tokens
        return [tok.replace(stupid_char, '') for tok in tokens_raw]
    def load(fpath):
        model = Tokenizer.from_file(fpath)
        segmenter = SegmenterSentencePiece([])
        segmenter.model = model
        return segmenter
    def save(self, fpath):
        self.model.save(fpath)

class SegmenterWordPiece(Segmenter):
    def __init__(self, sents, vocab_size=1000, show_progress=False):
        self.model = Tokenizer(models.WordPiece())
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            show_progress=show_progress,
            special_tokens=['[UNK]'],
            continuing_subword_prefix='',
            end_of_word_suffix=''
        )
        self.model.train_from_iterator(sents, trainer=trainer)
    def segment(self, sent: str):
        return self.model.encode(sent).tokens
    def load(fpath):
        model = Tokenizer.from_file(fpath)
        segmenter = SegmenterWordPiece([])
        segmenter.model = model
        return segmenter
    def save(self, fpath):
        self.model.save(fpath)

def viterbi_segment(text, score_fn):
    """Find the best segmentation of the string of characters, given the
    UnigramTextModel P."""
    # best[i] = best probability for text[0:i]
    # words[i] = best word ending at position i
    P = {}
    n = len(text)
    words = [''] + list(text)
    best = [1.0] + [0.0] * n
    ## Fill in the vectors best, words via dynamic programming
    for i in range(n+1):
        for j in range(0, i):
            w = text[j:i]
            if w not in P:
                P[w] = score_fn(w)
            if P[w] * best[i - len(w)] >= best[i]:
                best[i] = P[w] * best[i - len(w)]
                words[i] = w
    ## Now recover the sequence of best words
    sequence = []; i = len(words)-1
    while i > 0:
        sequence[0:0] = [words[i]]
        i = i - len(words[i])
    ## Return sequence of best words and overall probability
    return sequence, best[-1]

"""
class Viterbi:
    def __init__(self, seq, score_fn, print_progress=False):
        self.seq = seq
        self.score_fn = score_fn
        self.best_path = None
        self.best_score = 0
        self.print_progress = print_progress
        self.P = {}
        for i in range(len(self.seq)):
            for j in range(i+1, len(self.seq)+1):
                subseq = ''.join(self.seq[i:j])
                if subseq not in self.P:
                    score = self.score_fn(subseq)
                    self.P[subseq] = score
        self._search([0], None)
                
    def format_path(self):
        return [''.join(self.seq[i:j]) for i,j in zip(self.best_path[:-1], self.best_path[1:])]
    
    def _search(self, seq_so_far, score_so_far):
        if seq_so_far[-1] == len(self.seq):
            # end state
            if self.best_path is None or score_so_far > self.best_score:
                self.best_path = seq_so_far
                self.best_score = score_so_far
                if self.print_progress:
                    print(
                        self.format_path(), 
                        self.best_path, 
                        self.best_score)
        else:
            # continue recursively
            j_start = seq_so_far[-1]+1
            for j in range(j_start, len(self.seq)+1):
                subseq = ''.join(self.seq[seq_so_far[-1]: j])
                new_score = self.P[subseq]
                if score_so_far is not None:
                    new_score += score_so_far
                self._search(seq_so_far + [j], new_score)



### Bellman K-Segmentation
def prepare_ksegments(series,weights):
    '''
    '''
    N = len(series)
    #
    wgts = np.diag(weights)
    wsum = np.diag(weights*series)
    sqrs = np.diag(weights*series*series)

    dists = np.zeros((N,N))
    means = np.diag(series)

    for i in range(N):
        for j in range(N-i):
            r = i+j
            wgts[j,r] = wgts[j,r-1] + wgts[r,r]
            wsum[j,r] = wsum[j,r-1] + wsum[r,r]
            sqrs[j,r] = sqrs[j,r-1] + sqrs[r,r]
            means[j,r] = wsum[j,r] / wgts[j,r]
            dists[j,r] = sqrs[j,r] - means[j,r]*wsum[j,r]

    return dists, means

def regress_ksegments(series, weights, k):
    '''
    '''
    N = len(series)

    dists, means = prepare_ksegments(series, weights)

    k_seg_dist = np.zeros((k,N+1))
    k_seg_path = np.zeros((k,N))
    k_seg_dist[0,1:] = dists[0,:]

    k_seg_path[0,:] = 0
    for i in range(k):
        k_seg_path[i,:] = i

    for i in range(1,k):
        for j in range(i,N):
            choices = k_seg_dist[i-1, :j] + dists[:j, j]
            best_index = np.argmin(choices)
            best_val = np.min(choices)

            k_seg_path[i,j] = best_index
            k_seg_dist[i,j+1] = best_val

    reg = np.zeros(series.shape)
    rhs = len(reg)-1
    for i in reversed(range(k)):
        lhs = k_seg_path[i,rhs]
        reg[int(lhs):rhs] = means[int(lhs),rhs]
        rhs = int(lhs)

    return reg
                """