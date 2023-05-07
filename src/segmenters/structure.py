from typing import Iterator, List, Type, Union, Dict, Tuple
import numpy as np
import itertools
import collections
import os, sys
import pickle
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
    def __init__(self, tokens:List=[], set_last_type_as_unk:bool=False, dont_do_nothing:bool=False):
        if not dont_do_nothing: 
            types, counts = np.unique([t for t in tokens], return_counts=True)
            self.idx_to_word = list(types)
            self.idx_to_count = list(map(int, counts))
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
    def __init__(self, idx_to_word:List, word_to_count:Dict, sequences:Iterator, dirname:str=None):
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

    def _build(fpath, dirname, unk_token=default_unk_token, in_memory=True):
        # create the destination for the corpus files
        if not os.path.isdir(dirname): os.mkdir(dirname)

        # iterators over the text
        #tfiter = TextFileIterator(filenames=[fpath])
        tfiter = TryFromFile(fpath)
        #itertokens = lambda: map(Corpus.split_line, tfiter)
        itertokens = it.RestartableMapIterator(tfiter, Corpus.split_line)

        print("Building vocabulary...")
        #itertokens_flat = lambda: itertools.chain.from_iterable(itertokens())
        itertokens_flat = itertools.chain.from_iterable(itertokens)
        word_to_count = dict(collections.Counter(itertokens_flat).most_common())
        idx_to_word = list(word_to_count.keys())
        idx_to_word.append(unk_token)
        word_to_idx = dict(zip(idx_to_word, np.arange(len(idx_to_word))))

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
                idx = [word_to_idx[token] for token in tokens]
                if in_memory:
                    sequences.append(idx)
                line = ' '.join(map(str, idx)) + '\n'
                f.write(line)        
        return idx_to_word, word_to_count, sequences if in_memory else sequences_fpath, dirname

    def build(fpath, dirname, unk_token=default_unk_token, in_memory=True):
        return Corpus(*Corpus._build(fpath, dirname, unk_token, in_memory))

    def split_line(line):
        return [subw for word in line.split() for subw in word.split('-')]

class InvalidSegmentation(Exception):
    pass

class StructuredCorpus(Corpus):
    def __init__(self, idx_to_word:List, word_to_count:Dict, sequences:Iterator, dirname:str=None):
        super().__init__(idx_to_word, word_to_count, sequences, dirname)
        self.segmentations = []
        if not os.path.isdir(self._seg_dir()):
            os.mkdir(self._seg_dir())

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
                    raise TypeError('key must be a string or an iterable (of segmentation name(s))')
            segmentations = [self.get_segmentation(key) for key in keys]
            segmentations = [self._seg_fpath(key) if seg is None else seg for key, seg in zip(keys, segmentations)]
            sequences = self.sequences

        # sval is now either a path to a segmentation or a segmentation in memory represented by a list
        return SegmentationAligner(
            segmentations=segmentations, 
            sequences=sequences)             

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
            sval = None
            if self.in_memory:
                fpath = self._seg_fpath(sname)
                sval = []
                with open(fpath, 'r') as f:
                    for line in f:
                        labels = line.strip().split()
                        if len(labels) == 1:
                            sval.append(StructuredCorpus._read_label(labels[0]))
                        else:
                            sval.append([StructuredCorpus._read_label(l) for l in labels])           
            self.add_segmentation((sname, sval))
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
        if len(slist) != len(self.sequences):
            raise InvalidSegmentation("length of the segmentation must match the length of the sequences")
        if type(sname) != str or len(sname)==0:
            sname = 'anonymous_segmentation'
        sfpath = self._seg_fpath(sname)
        delete = False
        with open(sfpath, 'w') as f:
            if type(slist[0]) in [int, str]:
                for label in slist:
                    f.write(str(label) + '\n')
            elif type(slist[0]) == list:
                for i, (labels, seq) in enumerate(zip(slist, self.sequences)):
                    if len(labels) != len(seq):
                        delete = True
                        raise InvalidSegmentation(f"length of segmentation does not match at line {i} ({len(seq)}!={len(labels)})")
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

    def build(fpath:str, dirname:str, unk_token:str=default_unk_token, in_memory:bool=True):
        """ 'segmentations' is a list of tuples (sname, slist) where 'slist' is either a list of labels or list of lists of labels. """
        corpus_attributes = Corpus._build(fpath, dirname, unk_token, in_memory)
        corpus = StructuredCorpus(*corpus_attributes)
        default_seg = [i for i, _ in enumerate(corpus)]
        segmentations = [('default', default_seg)]
        for seg in segmentations:
            corpus.add_segmentation(seg, overwrite=True)
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

    def derive_segment_boundaries(self, sname_coarse:Key, sname_fine:Key=None):
        """Yields (boundary) indices of sname_coarse with respect to sname_fine. Useful to preserve segmentation info when transforming/summarizing the data under the fine segmentation.

        Args:
            sname_coarse (str|list[str]): Name/key(s) of the coarse segmentation.
            sname_fine (str|list[str], optional): Name/key(s) of the fine segmentation. If none, then original tokens will be used. Defaults to None.

        Yields:
            _type_: _description_
        """
        aligner_coarse = self[sname_coarse]
        sname_combined = StructuredCorpus.combine_key(sname_coarse, sname_fine)
        aligner_fine = self[sname_combined] # use both to ensure all segment boundaries in the coarse one are also in the fine one
        iter_fine = iter(aligner_fine)
        last_boundary = None
        for segment_coarse, label_coarse in aligner_coarse:
            # count how many fine segments until an equal chunk with segment_coarse is reached
            if last_boundary is not None: # yield with iteration delay, so that the last boundary (end of corpus) is implicit
                yield last_boundary
            chunk = []
            counter = 0
            while chunk != segment_coarse: 
                counter += 1
                segment_fine, label_fine = next(iter_fine)
                chunk += segment_fine

            boundary = counter - 1 if last_boundary is None else last_boundary + counter
            last_boundary = boundary 

class TryFromFile:
    def __init__(self, iterable):
        self.is_file = False
        if hasattr(iterable, '__iter__'): 
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

        line_lens = [len(line) for line in lines if type(line)==list]
        if len(line_lens) > 0: 
            # need to expand
            if len(set(line_lens)) > 1: 
                raise Exception("segmentations arent aligned!")
            max_len = max(line_lens)
            expanded = [line if type(line) == list else list(expand_to(line, max_len)) for line in lines]
            return expanded
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

    def __iter__(self):
        self.seg_iter = iter(self.seg_parser)
        if not self.seg_iter.is_nested:
            self.seq_iter = iter(map(TryFromIterable.readline, TryFromFile(self.sequences)))
        else:
            self.seq_iter = iter(TryFromIterable(self.sequences))
        return self
        
    def __next__(self):
        n, label = next(self.seg_iter)
        #try:
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
        #except StopIteration:
        #    raise InvalidSegmentation("segmentation and sequence are not aligned")
        return subseq, label

""" Some tests: 
aligned = SegmentationAligner([1,1,2,2,2], [1,2,3,4,5])
print(list(aligned))
"""
"""
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