import numpy as np
import itertools
import collections
import os
import pickle 
import iterator as it 

"""
class Vocab():
    def __init__(self, sentences, add_special_tokens=True):
        self.counter = collections.Counter(itertools.chain.from_iterable(sentences))
        #distinct_words = set()
        #for sentence in sentences:
        #    sent_list = [word for word in sentence] # if sentence was a string (not a list) convert it to a list
        #    distinct_words = distinct_words.union(sent_list)
        self.idx_to_word = list(sorted(self.counter.keys()))
        self.size = len(self.idx_to_word)
        self.word_to_idx = dict(zip(self.idx_to_word, np.arange(self.size)))
        self.special_tokens=add_special_tokens
        if self.special_tokens:
            self.w_unk = self.size; self.idx_to_word.append('<UNK>')
            self.w_mask = self.size+1; self.idx_to_word.append('<MASK>')
            self.w_start = self.size+2; self.idx_to_word.append('<START>')
            self.w_end = self.size+3; self.idx_to_word.append('<END>')

    def __len__(self):
        if self.special_tokens:
            return self.size+4
        else:
            return self.size

    def to_idx(self, sequences_words, special_tokens=True):
        if self.special_tokens and special_tokens:
            return [
                ([self.w_start]+[self.word_to_idx.get(w, self.w_unk) for w in seq]+[self.w_end])
                for seq in sequences_words]
        else:
            return [
                [self.word_to_idx[w] for w in seq if w in self.word_to_idx] 
                for seq in sequences_words]

    def to_words(self, sequences_idx):
        return [[self.idx_to_word[idx] for idx in seq] for seq in sequences_idx]

    def to_idx_masked(self, sequence_idx, return_masked=False):
        idx = [sequence_idx for i in range(len(sequence_idx))]
        idx = np.array(idx)
        masked = idx[np.diag_indices_from(idx)]
        idx[np.diag_indices_from(idx)] = self.w_mask
        if return_masked:
            return idx, masked
        else:
            return idx
"""

class NoMoreCorpusFiles(Exception):
    pass

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
    def __init__(self, tokens=[], set_last_type_as_unk=False, dont_do_nothing=False):
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
    def __init__(self, idx_to_word, word_to_count, sequences, dirname=None):
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
    def __init__(self, idx_to_word, word_to_count, sequences, dirname=None):
        super().__init__(idx_to_word, word_to_count, sequences, dirname)
        self.segmentations = []
        if not os.path.isdir(self._seg_dir()):
            os.mkdir(self._seg_dir())

    def __getitem__(self, key):
        """ Returns an iterable of a segmentation with name 'key' """
        if type(key) == str: 
            keys = [key]
        elif type(key) == list: 
            keys = key
        else: 
            raise TypeError('key must be a string or a list (of segmentation name(s))')

        segmentations = [self.get_segmentation(key) for key in keys]
        segmentations = [self._seg_fpath(key) if seg is None else seg for key, seg in zip(keys, segmentations)]
        # sval is now either a path to a segmentation or a segmentation in memory represented by a list
        return SegmentationAligner(
            segmentations=segmentations, 
            sequences=self.sequences)               

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
        snames = [fpath.split('/')[-1].split('.')[0] for fpath in seg_fpaths]
        for sname in snames:
            try:
                self.get_segmentation(sname)
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

    def add_segmentation(self, seg, overwrite=False):
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

    def build(fpath, dirname, unk_token=default_unk_token, in_memory=True):
        """ 'segmentations' is a list of tuples (sname, slist) where 'slist' is either a list of labels or list of lists of labels. """
        corpus_attributes = Corpus._build(fpath, dirname, unk_token, in_memory)
        corpus = StructuredCorpus(*corpus_attributes)
        default_seg = [i for i, _ in enumerate(corpus)]
        segmentations = [('default', default_seg)]
        for seg in segmentations:
            corpus.add_segmentation(seg, overwrite=True)
        return corpus

    def load(dirname, in_memory=True):
        corpus_attributes = Corpus._load(dirname, in_memory)
        corpus = StructuredCorpus(*corpus_attributes)
        corpus.load_segmentations()
        return corpus

class TryFromFile:
    def __init__(self, iterable):
        self.iterable = iterable
        if not self.in_memory() and not os.path.isfile(iterable):
            raise FileNotFoundError("must provide a valid file path to the iterable")
    def in_memory(self):
        return type(self.iterable)==list
    def __iter__(self):
        if not self.in_memory():
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
        singlify = lambda vals: vals[0] if len(vals) == 1 else vals
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

class SegmentationParser:
    def __init__(self, segmentations):
        self.segmentations = [TryFromIterable(segmentation) for segmentation in segmentations]
        self.seg_iters = None

    def do_next(self): 
        assert self.seg_iters is not None
        label, is_nested = zip(*[next(seg_iter) for seg_iter in self.seg_iters])
        if len(set(is_nested)) > 1: 
            raise Exception("cant combine per-line and per-word segmentations") # TODO do such that you can 
        if len(label) == 1: 
            label = label[0] # just make it a string if theres only one segmentation; else it will be a tuple of strings
        return label, is_nested[0]

    def __iter__(self):
        self.seg_iters = [iter(segmentation) for segmentation in self.segmentations]
        self.reached_end = False
        try:
            self.label_start, self.is_nested = self.do_next()
        except StopIteration:
            self.reached_end=True
        return self
        
    def __next__(self):
        if self.reached_end:
            raise StopIteration()
        # read labels until label changes
        n = 1
        current_label = self.label_start
        while True:
            try:
                self.label_start, _ = self.do_next()
                if current_label == self.label_start:
                    n += 1
                    continue
                break                    
            except StopIteration:
                self.reached_end = True
                break
        return n, current_label

class SegmentationAligner:
    def __init__(self, segmentations, sequences):
        self.seg_parser = SegmentationParser(segmentations)
        self.sequences = sequences

    def __iter__(self):
        self.seg_iter = iter(self.seg_parser)
        self.whole_lines = not self.seg_iter.is_nested
        if self.whole_lines:
            self.seq_iter = iter(map(TryFromIterable.readline, TryFromFile(self.sequences)))
        else:
            self.seq_iter = iter(TryFromIterable(self.sequences))
        return self
        
    def __next__(self):
        n, label = next(self.seg_iter)
        try:
            subseq = []
            for _ in range(n):
                if self.whole_lines:
                    val = next(self.seq_iter)
                    if type(val) == list:
                        subseq += val
                    else:
                        subseq.append(val)
                else:
                    subseq.append(next(self.seq_iter)[0])
        except StopIteration:
            raise InvalidSegmentation("segmentation and sequence are not aligned")
        return subseq, label

""" Some tests:
aligned = SegmentationAligner([1,1,2,2,2], [1,2,3,4,5])
print(list(aligned))
aligned = SegmentationAligner(sequences=[[1,2,1,2,1], [3,4,3], [6,7,6,7]], segmentation=[1,2,3])
print(list(aligned))
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