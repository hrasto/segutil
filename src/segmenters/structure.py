from __future__ import annotations
from posixpath import split
from random import sample
import random
import shutil
from signal import raise_signal
from tkinter.tix import Tree
from typing import Iterator, List, Type, Union, Dict, Tuple
import numpy as np
from numpy.random import default_rng
import itertools
import collections
import os, sys
import pickle
from segmenters.iterator import RestartableMapIterator, MaskedIterator, line2characters, line2subwords, line2words, line2characters_whitespace

try: 
    from . import iterator as it
except ImportError: 
    import iterator as it

class NoMoreCorpusFiles(Exception):
    pass

Key = Union[str, Iterator[str]]

class TextFileIterator:
    def __init__(self, corpus_dir="", filenames=[]):
        self.filenames = []
        try:
            self.filenames = [os.path.join(corpus_dir, fname) for fname in os.listdir(corpus_dir)]
        except FileNotFoundError:
            pass
        self.filenames += [fname for fname in filenames if os.path.isfile(fname)]
        self.filenames = sorted(self.filenames)
        if len(self.filenames) == 0: 
            raise Exception("Directory '{corpus_dir}' does not contain any files!")
        self._current_file = None
        
    def _set_current_file(self):
        if self._current_file is None: 
            self._current_filename_i = 0
        else:
            self._current_file.close()
            self._current_filename_i += 1

        if self._current_filename_i == len(self.filenames):
            raise NoMoreCorpusFiles()
        
        filename = self.filenames[self._current_filename_i]
        self._current_file = open(filename, "r")
        self._current_fileiter = iter(self._current_file) # returns lines from the file

    def __iter__(self):
        if self._current_file is not None:
            self._current_file.close()
            self._current_file = None
        self._set_current_file()
        return self

    def __next__(self):
        try: 
            return next(self._current_fileiter)
        except StopIteration as e:
            # current corpus file is finished, lets try to open a next corpus file
            pass

        try:
            self._set_current_file()
        except NoMoreCorpusFiles as e:
            raise StopIteration()

        return next(self._current_fileiter)

    def read_flat(self):
        return itertools.chain.from_iterable(map(str.split, self.__iter__()))

#def list_mirror(items):
#    return dict(zip(items, np.arange(len(items))))

default_unk_token = '<UNK>'

class Vocab:
    idx_to_word: List
    idx_to_count: List
    word_to_idx: dict

    def __init__(self, tokens:List=[], set_last_type_as_unk:bool=False, dont_do_nothing:bool=False, max_vocab_size:int=100000):
        if not dont_do_nothing: 
            #types, counts = np.unique([t for t in tokens], return_counts=True) # returns types in ascending order 
            #self.idx_to_word = list(types)
            #self.idx_to_count = list(map(int, counts))
            ct = collections.Counter([t for t in tokens])
            self.idx_to_word, self.idx_to_count = zip(*ct.most_common(max_vocab_size))
            
        self.word_to_idx = {word: idx for idx, word in enumerate(self.idx_to_word)}

        if set_last_type_as_unk:
            self.unk_token = self.idx_to_word[-1]
        else:
            self.unk_token = self.add_type(default_unk_token)

    def __len__(self):
        return len(self.idx_to_word)

    def add_type(self, token, count=0):
        self.word_to_idx[token] = len(self)
        self.idx_to_word.append(token)
        self.idx_to_count.append(count)
        return token

    def token_count(self):
        return int(sum(self.idx_to_count))

    def encode_token(self, token):
        try:
            return self.word_to_idx[token]
        except KeyError:
            return self.word_to_idx[self.unk_token]
    
    def encode_sent(self, sent):
        return [self.encode_token(token) for token in sent]

    def encode_batch(self, sents):
        return [self.encode_sent(sent) for sent in sents]

    def decode_token(self, token):
        return self.idx_to_word[token]

    def decode_sent(self, sent, stringify=False):
        tokens = [self.decode_token(token) for token in sent]
        if stringify: 
            return ' '.join(tokens)
        else: 
            return tokens

    def decode_batch(self, sents, stringify=False):
        return [self.decode_sent(sent, stringify) for sent in sents]

    def type_token_ratio(self):
        return len(self) / self.token_count()

    def type_lengths(self):
        return list(map(len, self.idx_to_word))

    def avg_type_len(self):
        return float(np.mean(self.type_lengths()))

    def avg_token_len(self):
        return float(np.average(self.type_lengths(), weights=self.idx_to_count))

    def nll(self):
        proba = [(count+1) / (self.token_count()+len(self)) for count in self.idx_to_count] # add1 smoothing
        proba = np.array(proba)
        log_proba = np.log2(proba)
        return float(-np.sum(proba * log_proba))

    def statistics(self):
        res = {
            'token_count': self.token_count(),
            'type_count': len(self),
            'type_token_ratio': self.type_token_ratio(),
            'avg_type_len': self.avg_type_len(),
            'avg_token_len': self.avg_token_len(),
            'nll': self.nll(),
        }
        res['nllpc'] = res['nll'] / res['avg_token_len']
        return res

class Corpus(Vocab):
    dirname: str
    word_to_count: dict
    in_memory: bool
    sequences: Union[Iterator, str]

    def __init__(self, idx_to_word:List, word_to_count:Dict, sequences:Union[Iterator, str], dirname:str=None):
        """Initializes a corpus instance given a vocabulary and sequences. This constructor gets the metrics already prepared from the load/build function, so that vocab does not have to be recomputed every time we load a corpus.

        Args:
            idx_to_word (list): list of types
            word_to_count (dict): type->count
            sequences (iterable[iterable[int]]): encoded corpus (iterable of iterables of ints)
            dirname (str, optional): Corpus directory. Defaults to None.
        """
        self.dirname = dirname
        self.idx_to_word = idx_to_word
        self.word_to_count = word_to_count
        self.idx_to_count = [self.word_to_count[word] if word in self.word_to_count else 0 for word in self.idx_to_word]
        self.sequences = sequences
        self.in_memory = type(self.sequences) != str
        self.filehandle = None
        super().__init__(set_last_type_as_unk=True, dont_do_nothing=True) # assuming previously vocab was created such that last type was assigned as <unk> 

    def _parse_line(line):
        return [int(el) for el in line.strip().split()]

    def __iter__(self):
        if type(self.sequences) == str:
            self.filehandle = open(self.sequences, 'r')
            self.iter = map(Corpus._parse_line, self.filehandle)
        else:
            self.iter = iter(self.sequences)
        return self

    def iter_decoded(self):
        return it.RestartableMapIterator(self, lambda sent: self.decode_sent(sent))

    def __next__(self):
        try:
            return next(self.iter)
        except StopIteration:
            if self.filehandle is not None:
                self.filehandle.close()
            raise StopIteration

    def to_text(self, n=None):
        if n is None: 
            return '\n'.join([' '.join(line) for line in self.iter_decoded()])
        elif type(n) == int:
            return '\n'.join([' '.join(line) for _, line in zip(range(n), self.iter_decoded())])
        raise TypeError("n must be a string or int")

    def _load(dirname, in_memory=True):
        with open(os.path.join(dirname, 'idx_to_word.pkl'), 'rb') as f:
            idx_to_word = pickle.load(f)
        with open(os.path.join(dirname, 'word_to_count.pkl'), 'rb') as f:
            word_to_count = pickle.load(f)
        sequences = os.path.join(dirname, 'sequences.txt')
        sequences = os.path.abspath(sequences)
        if in_memory:
            with open(sequences, 'r') as f:
                sequences = [Corpus._parse_line(line) for line in f]
        return idx_to_word, word_to_count, sequences, dirname

    def load(dirname, in_memory=True):
        return Corpus(*Corpus._load(dirname, in_memory))

    def _build(lines:Iterator, dirname:str, unk_token:str=default_unk_token, in_memory=True, extra_tokens:List[str]=[], split_line:Union[str, any]=line2subwords, reference:Corpus=None, max_vocab_size:int=100000):
        # create the destination for the corpus files
        if not os.path.isdir(dirname): os.mkdir(dirname)

        # iterators over the text
        #tfiter = TextFileIterator(filenames=[fpath])
        #tfiter = TryFromFile(fpath)
        #itertokens = lambda: map(Corpus.split_line, tfiter)
        if type(lines) != list: 
            if type(split_line) == str:
                if split_line in ['w', 'word', 'words']: 
                    split_line = line2words
                elif split_line in ['sw', 'subword', 'subwords']: 
                    split_line = line2subwords
                elif split_line in ['ch', 'char', 'chars', 'character', 'characters']: 
                    split_line = line2characters
                else: 
                    split_line = lambda x: x
            itertokens = it.RestartableMapIterator(lines, split_line)
        else: 
            itertokens = lines # should be a nested list in this case 

        #itertokens_flat = lambda: itertools.chain.from_iterable(itertokens())
        if reference is None: 
            print("Building vocabulary...")
            itertokens_flat = itertools.chain.from_iterable(itertokens)
            word_to_count = dict(collections.Counter(itertokens_flat).most_common(max_vocab_size))
            idx_to_word = list(word_to_count.keys())
            idx_to_word.append(unk_token)
            for token in extra_tokens:
                if token not in idx_to_word:
                    idx_to_word.append(token)
            word_to_idx = dict(zip(idx_to_word, range(len(idx_to_word))))
        else: 
            word_to_idx = reference.word_to_idx
            idx_to_word = reference.idx_to_word
            word_to_count = reference.word_to_count
        word_to_count[unk_token]=0

        with open(os.path.join(dirname, 'idx_to_word.pkl'), 'wb') as f:
            pickle.dump(idx_to_word, f)
        with open(os.path.join(dirname, 'word_to_count.pkl'), 'wb') as f:
            pickle.dump(word_to_count, f)

        print("Encoding tokens...")
        sequences = []
        sequences_fpath = os.path.join(dirname, 'sequences.txt')
        sequences_fpath = os.path.abspath(sequences_fpath)
        with open(sequences_fpath, 'w') as f:
            for tokens in itertokens:
                if reference is None: 
                    idx = []
                    for token in tokens:
                        try:
                            idx.append(word_to_idx[token])
                        except KeyError: 
                            idx.append(word_to_idx[unk_token])
                            word_to_count[unk_token] += 1
                else: 
                    idx = reference.encode_sent(tokens)

                if in_memory:
                    sequences.append(idx)
                line = ' '.join(map(str, idx)) + '\n'
                f.write(line)        
        
        if in_memory: 
            return idx_to_word, word_to_count, sequences, dirname
        else: 
            return idx_to_word, word_to_count, sequences_fpath, dirname

    def build(fpath:Union[str, Iterator], dirname:str, unk_token:str=default_unk_token, in_memory:bool=True, extra_tokens:List[str]=[], split_line:Union[str, any]=line2subwords, reference: Corpus=None, max_vocab_size:int=100000):
        if type(fpath) != list:
            lines = TryFromFile(fpath)
        else: 
            lines = fpath
        return Corpus(*Corpus._build(lines, dirname, unk_token, in_memory, extra_tokens, split_line, reference, max_vocab_size))

    def make_splits(self, split_size: Union[int, float], sample_size: Union[int, float]=1.0, seed=None, randomize=False) -> Tuple[np.ndarray, np.ndarray]:
        rng = default_rng(seed)
        data_len = sum([1 for _ in iter(self)])

        # in case sample size is provided, only consider a random subset of the training data
        sample_size_int = sample_size if type(sample_size) == int else int(sample_size*data_len)
        sample_size_int = min(data_len, max(sample_size_int, 0))
        mask_general = None
        if sample_size_int < data_len: 
            if randomize: 
                sample_idx = rng.choice(a=data_len, size=sample_size_int, replace=False)
            else: 
                sample_idx = np.arange(sample_size_int)
        else: 
            sample_idx = np.arange(data_len)

        # make train/test splits via a mask
        split_size_int = split_size if type(split_size) == int else int(split_size*sample_size_int)
        if split_size_int < 0: 
            split_size_int += sample_size_int
        else: 
            split_size_int = min(sample_size_int, split_size_int)
            
        split_idx = sample_idx[:split_size_int]
        other_idx = sample_idx[split_size_int:]

        # convert indices to masks
        split_mask = np.zeros(data_len) == 1
        split_mask[split_idx] = True
        other_mask = np.zeros(data_len) == 1
        other_mask[other_idx] = True

        return split_mask, other_mask

    def split(self, mask_train: np.ndarray, mask_test: np.ndarray) -> Tuple[Corpus, Corpus]:
        #mask_train, mask_test = self.make_splits(split_size, sample_size, seed)
        #print(f'total size {data_len}; sample size {sample_size_int}; split size {split_size_int}')
        sequences_train = list(MaskedIterator(self, mask_train))
        sequences_test = list(MaskedIterator(self, mask_test))
        corpus_train = Corpus(self.idx_to_word.copy(), self.word_to_count.copy(), sequences_train)
        corpus_test = Corpus(self.idx_to_word.copy(), self.word_to_count.copy(), sequences_test)
        return corpus_train, corpus_test

    def save(self, dirname:str=None) -> Corpus:
        if dirname is not None:
            self.dirname = dirname
        if not os.path.isdir(self.dirname):
            os.mkdir(self.dirname)
        with open(os.path.join(self.dirname, 'sequences.txt'), 'w') as f: 
            for seq in self: 
                f.write(' '.join([str(_) for _ in seq]) + '\n')
        with open(os.path.join(self.dirname, 'idx_to_word.pkl'), 'wb') as f:
            pickle.dump(self.idx_to_word, f)
        with open(os.path.join(self.dirname, 'word_to_count.pkl'), 'wb') as f:
            pickle.dump(self.word_to_count, f)

        return self

class InvalidSegmentation(Exception):
    pass

class StructuredCorpus(Corpus):
    def __init__(self, idx_to_word:List, word_to_count:Dict, sequences:Iterator, dirname:str=None):
        super().__init__(idx_to_word, word_to_count, sequences, dirname)
        self.segmentations = []
        if not os.path.isdir(self._seg_dir()):
            os.mkdir(self._seg_dir())
        self.postprocess=lambda x: x

    def __getitem__(self, keys:Key):
        """Returns an iterable of a segmentation with name 'key' if key is string, or a union of segmentations if key is iterable.

        Args:
            key (Union[str, List[str]]): string or list of strings of keys. 

        Raises:
            TypeError: If key is an idiot

        Returns:
            SegmentationAligner: iterable yielding tuples of (segment, label) where segment is a list of tokens
        """
        if keys is None: 
            segmentations = [range(sys.maxsize)]
            sequences = it.RestartableFlattenIterator(self.sequences)
        else:
            if type(keys) == str: 
                keys = [keys]
            else: 
                try: 
                    keys = [_ for _ in keys]
                except: 
                    raise TypeError('key must be either a string or an iterable over strings')
            segmentations = [self.get_segmentation(key) for key in keys]
            segmentations = [self._seg_fpath(key) if seg is None else seg for key, seg in zip(keys, segmentations)]
            sequences = self.sequences

        # sval is now either a path to a segmentation or a segmentation in memory represented by a list
        return SegmentationAligner(
            segmentations=segmentations, 
            sequences=sequences).postprocess(self.postprocess)

    def save(self, dirname: str = None, copy_segmentations:bool=False) -> StructuredCorpus:
        prev_seg_dirname = self._seg_dir()
        super().save(dirname) # creates a directory, saves vocab and sequences
        # now save segmentations
        segs_dirname = self._seg_dir()
        if not os.path.isdir(segs_dirname):
            os.mkdir(segs_dirname)
        if copy_segmentations and prev_seg_dirname is not None and os.path.isdir(prev_seg_dirname): 
            for sname in self.get_segmentation_names():
                fpath_prev = os.path.join(prev_seg_dirname, sname+'.txt')
                fpath_new = self._seg_fpath(sname)
                shutil.copy(fpath_prev, fpath_new)

        return self

    def split_segmentation(self, sname:str, mask_split: np.ndarray, mask_rest: np.ndarray) -> Tuple[Iterator, Iterator]:
        split_seg_iter = TryFromFile(self._seg_fpath(sname))
        split_seg_masked = MaskedIterator(split_seg_iter, mask_split)
        rest_seg_iter = TryFromFile(self._seg_fpath(sname))
        rest_seg_masked = MaskedIterator(rest_seg_iter, mask_rest)
        return split_seg_masked, rest_seg_masked

    def split(self, mask_split: np.ndarray, mask_rest: np.ndarray, name_split:str='train', name_rest:str='test') -> Tuple[StructuredCorpus, StructuredCorpus]:
        corpus_split, corpus_rest = super().split(mask_split, mask_rest)
        dirname_split = self.dirname+'_'+name_split
        dirname_rest = self.dirname+'_'+name_rest
        if not os.path.isdir(dirname_split): os.mkdir(dirname_split)
        if not os.path.isdir(dirname_rest): os.mkdir(dirname_rest)
        scorpus_split = StructuredCorpus(
            corpus_split.idx_to_word, 
            corpus_split.word_to_count, 
            corpus_split.sequences, 
            dirname_split).save()
        scorpus_rest = StructuredCorpus(
            corpus_rest.idx_to_word, 
            corpus_rest.word_to_count, 
            corpus_rest.sequences,
            dirname_rest).save()

        for sname in self.get_segmentation_names():
            seg_split, seg_rest = self.split_segmentation(sname, mask_split, mask_rest)
            scorpus_split.add_segmentation((sname, seg_split)) # these calls also store the segmentation file
            scorpus_rest.add_segmentation((sname, seg_rest))

        return scorpus_split, scorpus_rest

    def decode_segmented(self, sname: Key) -> List[Tuple[str, str]]:
        seg_aligner = self[sname]
        res = []
        for idx, label in seg_aligner: 
            sent = self.decode_sent(idx, True)
            res.append((sent, label))
        return res

    def iter_flat(self) -> Iterator:
        # return a restartable iterator
        return it.RestartableMapIterator(self[None], lambda x: x[0]) 

    def get_segmentation_names(self):
        snames, _ = zip(*self.segmentations)
        return list(snames)

    def get_segmentation(self, sname):
        for _sname, sval in self.segmentations:
            if _sname == sname:
                return sval
        raise KeyError(f"there is no segmentation with name '{sname}'")

    def _seg_dir(self):
        if self.dirname is None or not os.path.isdir(self.dirname):
            raise Exception(f"Valid existing corpus directory must be provided (got '{self.dirname}')")
        return os.path.join(self.dirname, "segmentation")

    def _seg_fpath(self, sname):
        return os.path.join(self._seg_dir(), f"{sname}.txt")

    def _read_label(label):
        if type(label) == str and label.isnumeric():
            return int(label)
        return label

    def load_segmentations(self):
        seg_fpaths = os.listdir(self._seg_dir())
        snames = [fpath.split('/')[-1].split('.')[0] for fpath in seg_fpaths if fpath[0] != '.' and fpath.split('.')[-1] == 'txt']
        for sname in snames:
            try:
                self.get_segmentation(sname) # skip if seg already loaded 
                continue
            except KeyError:
                pass
            slist = None
            if self.in_memory:
                fpath = self._seg_fpath(sname)
                slist = []
                with open(fpath, 'r') as f:
                    for line in f:
                        labels = line.strip().split()
                        if len(labels) == 1:
                            slist.append(StructuredCorpus._read_label(labels[0]))
                        else:
                            slist.append([StructuredCorpus._read_label(l) for l in labels])           
            #self.add_segmentation((sname, sval))
            try:
                idx = self.get_segmentation_names().index(sname)
                self.segmentations[idx] = (sname, slist if self.in_memory else None)
            except ValueError:
                self.segmentations.append((sname, slist if self.in_memory else None))
        return self

    def add_segmentation(self, seg:Tuple[str, Iterator], overwrite:bool=False):
        try:
            sname, slist = seg
        except TypeError:
            raise InvalidSegmentation("must provide a tuple of shape (sname, slist) per segmentation")
        append = True
        try:
            self.get_segmentation(sname)
            if not overwrite:
                raise Exception(f"segmentation with name '{sname}' already exists")
            append = False
        except KeyError:
            pass
        
        #if len(slist) != len(list(self)):
        #    raise InvalidSegmentation("length of the segmentation must match the length of the sequences")

        if type(sname) != str or len(sname)==0:
            sname = 'anonymous_segmentation'
        sfpath = self._seg_fpath(sname)
        delete = False
        first_entry = next(iter(slist))
        with open(sfpath, 'w') as f:
            if type(first_entry) in [int, str]:
                for label in slist:
                    f.write(str(label) + '\n')
            elif type(first_entry) == list:
                for i, (labels, seq) in enumerate(zip(slist, self.sequences)):
                    if len(labels) != len(seq):
                        delete = True
                        raise InvalidSegmentation(f"length of segmentation '{sname}' does not match at line {i} ({len(seq)}!={len(labels)})")
                    f.write(' '.join(str(label) for label in labels) + '\n')
            else:
                raise InvalidSegmentation("segmentation entry must be either a list, int or a string")
        if delete:
            os.remove(sfpath)
        try:
            idx = self.get_segmentation_names().index(sname)
            self.segmentations[idx] = (sname, slist if self.in_memory else None)
        except ValueError:
            self.segmentations.append((sname, slist if self.in_memory else None))

    def build(fpath:Union[str, Iterator], dirname:str, unk_token:str=default_unk_token, in_memory:bool=True, extra_tokens:List[str]=[], split_line:Union[str, any]=line2subwords, reference:Corpus=None, max_vocab_size:int=100000):
        """ 'segmentations' is a list of tuples (sname, slist) where 'slist' is either a list of labels or list of lists of labels. """
        if type(fpath) == list:
            lines = fpath
        else:
            lines = TryFromFile(fpath)

        corpus_attributes = Corpus._build(lines, dirname, unk_token, in_memory, extra_tokens, split_line, reference, max_vocab_size)
        corpus = StructuredCorpus(*corpus_attributes)
        
        # default segmentation enumerates lines
        default_seg = [i for i, _ in enumerate(corpus)]
        corpus.add_segmentation(('default', default_seg), overwrite=True)

        # in case we segment by characters, add word-level indices 
        if type(fpath) != list and split_line in ['ch', 'char', 'chars', 'character', 'characters', line2characters]: 
            lines = TryFromFile(fpath)
            word_segs = lambda line: [i for i, word in enumerate(line2subwords(line)) for _ in range(len(word))]
            word_segmentation = RestartableMapIterator(lines, word_segs)
            corpus.add_segmentation(('word', word_segmentation), overwrite=True)

        # in case we segment by subwords, add word-level indices 
        if type(fpath) != list and split_line in ['s', 'sw', 'subword', 'subwords', line2subwords]: 
            lines = TryFromFile(fpath)
            word_segs = lambda line: [i for i, word in enumerate(line2words(line)) for _ in range(len(line2subwords(word)))]
            word_segmentation = RestartableMapIterator(lines, word_segs)
            corpus.add_segmentation(('word', word_segmentation), overwrite=True)

        return corpus

    def load(dirname:str, in_memory:bool=True):
        corpus_attributes = Corpus._load(dirname, in_memory)
        corpus = StructuredCorpus(*corpus_attributes)
        corpus.load_segmentations()
        return corpus

    def combine_key(key1:Key, key2:Key):
        key = []

        if type(key1) == str or key1 is None:
            key.append(key1)
        elif hasattr(key1, '__iter__'): 
            key += key1

        if type(key2) == str or key2 is None:
            key.append(key2)
        elif hasattr(key2, '__iter__'): 
            key += key2
        
        for k in key: 
            if k is None: return None

        return key

    def keys_overlap(k1:Key, k2:Key)->bool:
        if type(k1) == str or k1 is None:
            if type(k2) == str or k2 is None: 
                return k1 == k2
            else: # k2 is iterable
                for subkey in k2: 
                    if subkey == k1: return True
                return False
        else: #k1 is iterable 
            if type(k2) == str or k2 is None: 
                for subkey in k1: 
                    if subkey == k2: return True
                return False
            else: # k1 and k2 are iterable
                for sk1 in k1: 
                    for sk2 in k2: 
                        if sk1 == sk2: return True
                return False

    def keys_identical(k1:Key, k2:Key)->bool:
        if type(k1) == str or k1 is None:
            if type(k2) == str or k2 is None: 
                return k1 == k2
            else: # k2 is iterable
                return set((k1,)) == set(k2)
        else: #k1 is iterable 
            if type(k2) == str or k2 is None: 
                return set((k2,)) == set(k1)
            else: # k1 and k2 are iterable
                return set(k1) == set(k2)

    def derive_segment_boundaries(self, sname_coarse:Key, sname_fine:Key=None) -> Iterator:
        """Yields (boundary) indices of sname_coarse with respect to sname_fine. Useful to preserve segmentation info when transforming/summarizing the data under the fine segmentation.

        Args:
            sname_coarse (str|list[str]): Name/key(s) of the coarse segmentation.
            sname_fine (str|list[str], optional): Name/key(s) of the fine segmentation. If none, then original tokens will be used. Defaults to None.

        Yields:
            _type_: _description_
        """
        aligner_coarse = self[sname_coarse]

        if not StructuredCorpus.keys_overlap(sname_coarse, sname_fine):
            sname_fine = StructuredCorpus.combine_key(sname_coarse, sname_fine)

        aligner_fine = self[sname_fine] # use both to ensure all segment boundaries in the coarse one are also in the fine one
        iter_fine = iter(aligner_fine)
        last_boundary = None
        for segment_coarse, label_coarse in aligner_coarse:
            # count how many fine segments until an equal chunk with segment_coarse is reached
            if last_boundary is not None: # yield with iteration delay, so that the last boundary (end of corpus) is implicit
                yield last_boundary
            chunk = []
            counter = 0

            while len(chunk) < len(segment_coarse): 
                counter += 1
                segment_fine, label_fine = next(iter_fine)
                chunk += segment_fine
            
            if chunk != segment_coarse: 
                raise Exception('somethings wrong with the segmentation')

            boundary = counter - 1 if last_boundary is None else last_boundary + counter
            last_boundary = boundary 

    def segment_wrt(self, sname_coarse:Key, sname_fine: Key=None) -> Iterator:        
        aligner_coarse = self[sname_coarse]
        sname_combined = StructuredCorpus.combine_key(sname_coarse, sname_fine)
        aligner_fine = self[sname_combined] # use both to ensure all segment boundaries in the coarse one are also in the fine one
        iter_fine = iter(aligner_fine)
        last_boundary = None
        for segment_coarse, label_coarse in aligner_coarse:
            segments_fine = [] # nested
            chunk = [] # flat

            while len(chunk) < len(segment_coarse): 
                segment_fine, label_fine = next(iter_fine)
                chunk += segment_fine
                segments_fine.append((segment_fine, label_fine))
            
            if chunk != segment_coarse: 
                raise Exception('somethings wrong with the segmentation')

            yield segments_fine, label_coarse

class TryFromFile:
    def __init__(self, iterable: Union[Iterator, str]):
        self.is_file = False
        if hasattr(iterable, '__iter__') and type(iterable) != str: 
            pass # list/iterable, but not a string
        elif type(iterable)==str:
            if os.path.isfile(iterable):
                self.is_file = True
            else: 
                raise FileNotFoundError("must provide a valid file path to the iterable")
        else: 
            raise TypeError('argument iterable must be an iterable or a string containing a path to an existing text file')
        self.iterable = iterable

    def __iter__(self):
        if self.is_file:
            self.file = open(self.iterable, 'r')
            self.iter = iter(map(str.strip, self.file))
        else:
            self.file = None
            self.iter = iter(self.iterable)
        return self

    def __next__(self):
        try:
            return next(self.iter)
        except StopIteration:
            if self.file is not None: 
                self.file.close()
            raise StopIteration()

class TryFromIterable:
    def __init__(self, iterable):
        self.iterable = TryFromFile(iterable)

    def readline(line):
        if type(line) == int:
            return line
        
        if type(line) == str:
            line = line.strip().split()
            if len(line) > 1:
                return list(map(StructuredCorpus._read_label, line))
            if len(line) == 1:
                return StructuredCorpus._read_label(line[0])
            assert False

        if type(line) == list:
            return list(map(StructuredCorpus._read_label, line))
        assert False

    def __iter__(self):
        nested_labels_iterable = map(TryFromIterable.readline, self.iterable)
        self.iter = iter(nested_labels_iterable)
        self.sub_iter = None
        return self
        
    def __next__(self):
        if self.sub_iter is not None:
            try:
                return next(self.sub_iter), True
            except StopIteration:
                pass
        
        val = next(self.iter)
        if type(val) == str:
            return val, False
        try: 
            self.sub_iter = iter(val)
        except TypeError:
            return val, False
        # if code reaches here, self.sub_iter is an iterator
        return self.__next__()

def expand_to(x, n):
    for _ in range(n): 
        yield x

def is_at_least_one_nested(fnames_or_iterables, check_first_n_lines=10):
    for fi in fnames_or_iterables: 
        for _, raw_line in zip(range(check_first_n_lines), TryFromFile(fi)): 
            parsed_line = TryFromIterable.readline(raw_line)
            if type(parsed_line) == list: 
                return True
    return False

class SegmentationExpander:
    def __init__(self, fnames_or_iterables) -> None:
        self.is_nested = is_at_least_one_nested(fnames_or_iterables)
        self.fnames_or_iterables = fnames_or_iterables

    def __iter__(self):
        self.per_line_iter = [iter(TryFromFile(fi)) for fi in self.fnames_or_iterables]
        return self

    def __next__(self): 
        lines = []
        for pli in self.per_line_iter:
            line = next(pli)
            line_parsed = TryFromIterable.readline(line)
            lines.append(line_parsed)

        if self.is_nested:
            line_lens = [len(line) for line in lines if type(line)==list]
            if len(line_lens) > 0: 
                # need to expand
                if len(set(line_lens)) > 1: 
                    raise Exception("segmentations arent aligned!")
                max_len = max(line_lens)
                expanded = [line if type(line) == list else [line]*max_len for line in lines]
                return expanded
            else: 
                return [[line] for line in lines]
        else: 
            return tuple(lines)

class SegmentationParser:
    def __init__(self, segmentations):
        #self.segmentations = [TryFromIterable(segmentation) for segmentation in segmentations]
        expanded = SegmentationExpander(segmentations)
        self.is_nested = expanded.is_nested
        if self.is_nested: 
            fn = lambda labelset: list(zip(*labelset))
            self.segmentation = it.RestartableMapIterator(expanded, fn)
        else: 
            self.segmentation = expanded
        self.iter = None

    def __iter__(self):
        if self.is_nested: 
            # iterate over flattened version
            self.iter = iter(itertools.chain.from_iterable(self.segmentation))
        else: 
            self.iter = iter(self.segmentation)
        self.reached_end = False
        try:
            self.labels_start = next(self.iter)
        except StopIteration:
            self.reached_end=True
        return self
        
    def __next__(self):
        if self.reached_end:
            raise StopIteration()
        # read labels until label changes
        n = 1
        current_label = self.labels_start
        while True:
            try:
                self.labels_start = next(self.iter)
                if self.labels_start == ('d', 'R'):
                    print('here')
                if current_label == self.labels_start:
                    n += 1
                    continue
                break                    
            except StopIteration:
                self.reached_end = True
                break

        current_label = current_label if len(current_label) > 1 else current_label[0]
        return n, current_label

class SegmentationAligner:
    def __init__(self, segmentations, sequences):
        self.seg_parser = SegmentationParser(segmentations)
        self.sequences = sequences
        self.max_n=-1
        self.postprocess_fn=None

    def first_n(self, n:int):
        self.max_n=n
        return self

    def postprocess(self, fn):
        self.postprocess_fn = fn
        return self

    def __iter__(self):
        self.i=0
        self.seg_iter = iter(self.seg_parser)
        if not self.seg_iter.is_nested:
            self.seq_iter = iter(map(TryFromIterable.readline, TryFromFile(self.sequences)))
        else:
            self.seq_iter = iter(TryFromIterable(self.sequences))
        return self
        
    def __next__(self):
        if self.max_n != -1 and self.i == self.max_n:
            self.max_n = -1
            raise StopIteration()

        n, label = next(self.seg_iter)
        try:
            subseq = []
            for _ in range(n):
                if not self.seg_parser.is_nested:
                    val = next(self.seq_iter)
                    if type(val) == list:
                        subseq += val
                    else:
                        subseq.append(val)
                else:
                    subseq.append(next(self.seq_iter)[0])
        except StopIteration:
            #raise InvalidSegmentation("segmentation and sequence are not aligned")
            self.max_n=-1
            raise StopIteration()
        self.i += 1 
        if self.postprocess_fn is not None: 
            subseq = self.postprocess_fn(subseq)
        return subseq, label

"""
corpus = StructuredCorpus.load('../corpora_myformat/swda_train')
print(corpus.get_segmentation_names())
for b in corpus.derive_segment_boundaries('act_tag', 'pos'):
    print(b)
print(list(corpus[['pos', 'act_tag']].first_n(10)))
#print(list(corpus['act_tag'].first_n(10)))
aligned = SegmentationAligner([1,1,2,2,2], [1,2,3,4,5])
print(list(aligned))
tmp = 'tmp.txt'
dirname = 'applejuice'
corpus = StructuredCorpus.build(tmp, dirname)
corpus = StructuredCorpus.load(dirname)
#print(list(corpus.derive_segment_boundaries('default')))
print(list(corpus.decode_segmented('default')))
"""

""" Some tests: 
dirname = 'child_char'
fpath = '/Users/rastislavhronsky/ml-experiments/corpora_processed/child_proc_uniq_seg_CELEX_fbMFS_sub100.txt'
corpus=StructuredCorpus.build(fpath, dirname, split_line='ch')

splits = corpus.make_splits(10, 15, 1, True)
c1, c2 = corpus.split(*splits)
c1 = StructuredCorpus.load('child_train')

"""
"""
dirname = '../corpora_myformat/swda_test'
corpus=StructuredCorpus.load(dirname)
print(list(corpus.derive_segment_boundaries(['act_tag', 'default'], 'word'))[-5:])
print(len(list(corpus['word'])))
for segs_fine, lab_coarse in corpus.segment_wrt('default', 'word'):
    segs_fine, labs_fine = zip(*segs_fine)
    print(segs_fine, labs_fine, lab_coarse)
    print()
#print(list(corpus['word'].first_n(3).postprocess(lambda seq: ''.join(corpus.decode_sent(seq)))))
#corpus.save('child2', True)
for b in c1.derive_segment_boundaries('word'):
    print(b)
import random

corpus_name='010101'
segment_len=8
n_segments=100
# create dataset
#segmentation = [seg_i for seg_i in range(n_segments) for _ in range(segment_len)]
seq1 = [random.randint(0, 3) for i in range(n_segments*segment_len//2)]
seq2 = [random.randint(4, 7) for i in range(n_segments*segment_len-len(seq1))]
seq = list(itertools.chain.from_iterable(zip(seq1, seq2)))

fname = 'tmp.txt'
with open(fname,'w') as f: 
    for segment_i in range(n_segments):
        f.write(' '.join(map(str, seq[segment_len*segment_i:segment_len*(segment_i+1)]))+'\n')
corpus = Corpus.build(fname, dirname=corpus_name, in_memory=False)
#os.remove(fname)

c = StructuredCorpus.load('../corpora_myformat/test_structured')
iterator = c[None]

print(list(iterator))
print(list(iterator))
aligned = SegmentationAligner(sequences=[[1,2,1,2,1], [3,4,3], [6,7,6,7]], segmentations=[range(sys.maxsize)])
print(list(aligned))

c = StructuredCorpus.load('../corpora_myformat/test_structured')
print(list(c.derive_segment_boundaries('enum', None)))
aligned = SegmentationAligner(sequences=[[1,2,1,2,1], [3,4,3], [6,7,6,7]], segmentation=[1,2])
print(list(aligned))
aligned = SegmentationAligner(
    sequences=[[1,2,1,2,1], [3,4,3], [6,7,6,7]], 
    segmentation=[[1,1,1,2,2],[2,1,1],[1,2,2,2]])
print(list(aligned))
aligned = SegmentationAligner(sequences=[[1,2,1,2,1], [3,4,3], [6,7,6,7]], segmentation=[1,2])
print(list(aligned))
aligned = SegmentationAligner(
    sequences='doc2dial_segmentations/test_corpus.txt', 
    segmentation='doc2dial_segmentations/test_seg1.txt')
print(list(aligned))
aligned = SegmentationAligner(
    sequences='doc2dial_segmentations/test_corpus.txt', 
    segmentation='doc2dial_segmentations/test_seg2.txt')
print(list(aligned))
aligned = SegmentationAligner(
    sequences='doc2dial_segmentations/test_corpus.txt', 
    segmentation='doc2dial_segmentations/test_seg3.txt')
print(list(aligned))
"""

class DummyIter:
    def __iter__(self):
        return self
    def __next__(self):
        raise StopIteration()

class LMIter:
    def __init__(self, sequences_nested, context_size=5):
        self.sequences_nested = sequences_nested
        self.context_size = context_size

    def __iter__(self):
        self.iter=iter(self.sequences_nested)
        self.current_examples=DummyIter()
        return self
        
    def slide(seq, context_size):
        return [(seq[i:i+context_size], seq[i+context_size]) for i in range(len(seq)-context_size)]

    def __next__(self):
        try: 
            return next(self.current_examples)
        except StopIteration: 
            current_seq=next(self.iter) # if done, throws StopIteration
            self.current_examples=iter(LMIter.slide(current_seq, self.context_size))
            return self.__next__()

class Batchify:
    def __init__(self, iterable, batch_size):
        self.iterable = iterable
        self.batch_size = batch_size

    def __iter__(self):
        self.iter = iter(self.iterable)
        return self

    def __next__(self):
        res = []
        for _ in range(self.batch_size):
            try:
                res.append(next(self.iter))
            except StopIteration:
                break        
        if len(res) == 0:
            raise StopIteration()
        return res

def unzip_batch(batch):
    xs, ys = zip(*batch)
    return np.array(xs), np.array(ys)

def make_lm_dataset(corpus, context_size, batch_size):
    iter_corpus_lm = LMIter(corpus, context_size=context_size)
    iter_corpus_batched = map(unzip_batch, Batchify(iter_corpus_lm, batch_size=batch_size))
    return iter_corpus_batched