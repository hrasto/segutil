import numpy as np
import itertools
import collections
import os
import pickle

from build.lib.segmenters.structure import TryFromIterable 

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

class Vocab:
    def __init__(self, idx_to_word, last_token_unk=False):
        self.word_to_idx = {}
        self.idx_to_word = []
        for word in sorted(set(idx_to_word)):
            self.add_token(word)
        if last_token_unk:
            self.unk_token = self.idx_to_word[-1]
        else:
            self.unk_token = self.add_token('<UNK>')

    def __len__(self):
        return len(self.idx_to_word)

    def add_token(self, token):
        self.word_to_idx[token] = len(self.idx_to_word)
        self.idx_to_word.append(token)
        return token

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

    def decode_sent(self, sent):
        return [self.decode_token(token) for token in sent]

    def decode_batch(self, sents):
        return [self.decode_sent(sent) for sent in sents]

class Corpus(Vocab):
    def __init__(self, idx_to_word, word_to_count, sequences, dirname=None):
        super().__init__(idx_to_word, True)
        self.dirname = dirname
        self.word_to_count = word_to_count #word to count is without the UNK token!
        self.sequences = sequences
        self.in_memory = type(self.sequences) != str
        self.filehandle = None
        counts = np.array(list(word_to_count.values()))
        counts_rel = counts/counts.sum()
        nll = -np.log(counts_rel)
        self.word_to_nll = dict(zip(self.idx_to_word, nll))
        self.iter_decoded = False

    def _parse_line(line):
        return [int(el) for el in line.strip().split()]

    def __iter__(self):
        if type(self.sequences) == str:
            self.filehandle = open(self.sequences, 'r')
            self.iter = map(Corpus._parse_line, self.filehandle)
        else:
            self.iter = iter(self.sequences)
        return self

    def __next__(self):
        try:
            if self.iter_decoded:
                return self.decode_sent(next(self.iter))
            else:
                return next(self.iter)
        except StopIteration:
            if self.filehandle is not None:
                self.filehandle.close()
            raise StopIteration

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

    def _build(fpath, dirname, unk_token='<UNK>', in_memory=True):
        # create the destination for the corpus files
        if not os.path.isdir(dirname): os.mkdir(dirname)

        # iterators over the text
        #tfiter = TextFileIterator(filenames=[fpath])
        tfiter = TryFromFile(fpath)
        itertokens = lambda: map(Corpus.split_line, tfiter)
        itertokens_flat = lambda: itertools.chain.from_iterable(itertokens())

        print("Building vocabulary...")
        word_to_count = dict(collections.Counter(itertokens_flat()).most_common())
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
            for tokens in itertokens():
                idx = [word_to_idx[token] for token in tokens]
                if in_memory:
                    sequences.append(idx)
                line = ' '.join(map(str, idx)) + '\n'
                f.write(line)        
        return idx_to_word, word_to_count, sequences if in_memory else sequences_fpath, dirname

    def build(fpath, dirname, unk_token='<UNK>', in_memory=True):
        return Corpus(*Corpus._build(fpath, dirname, unk_token, in_memory))

    def split_line(line):
        return [subw for word in line.split() for subw in word.split('-')]

    def vocab_dim(self):
        """ includes the UNK """
        return len(self.idx_to_word) 

    def compute_nll(self):
        nll_ttl = 0
        for form, count, nll in zip(
            self.idx_to_word, 
            self.word_to_count.values(), 
            self.word_to_nll.values()):
            nll_ttl += count * nll
        return nll_ttl

    def n_tokens(self):
        return sum(self.word_to_count.values())

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
        sval = self.get_segmentation(key)
        if sval is None:
            sval = self._seg_fpath(key)
        # sval is now either a path to a segmentation or a segmentation in memory represented by a list
        return SegmentationAligner(
            segmentation=sval, 
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

    def build(fpath, dirname, unk_token='<UNK>', in_memory=True):
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

def read_corpus_line(line):
    word_alts = line.split()
    words = [wa.split('|')[0] for wa in word_alts]
    subwords = [sw for w in words for sw in w.split('-')]
    return subwords                

def read_segmentation_line(line):
    #singlify = lambda vals: vals[0] if len(vals) == 1 else vals
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

class FileOrIterableReader:
    def __init__(self, iterable, read_fn=read_segmentation_line):
        self.iterable = TryFromFile(iterable)
        self.read_fn = read_fn

    def __iter__(self):
        nested_labels_iterable = map(self.read_fn, self.iterable)
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
    def __init__(self, segmentation):
        self.segmentation = FileOrIterableReader(segmentation)

    def __iter__(self):
        self.seg_iter = iter(self.segmentation)
        self.reached_end=False
        try:
            self.label_start, self.is_nested = next(self.seg_iter)
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
                self.label_start = next(self.seg_iter)[0]
                if current_label == self.label_start:
                    n += 1
                    continue
                break                    
            except StopIteration:
                self.reached_end = True
                break
        return n, current_label

class SegmentationAligner:
    def __init__(self, segmentation, sequences):
        self.seg_parser = SegmentationParser(segmentation)
        self.sequences = sequences

    def __iter__(self):
        self.seg_iter = iter(self.seg_parser)
        self.whole_lines = not self.seg_iter.is_nested
        if self.whole_lines:
            self.seq_iter = iter(map(FileOrIterableReader.read_segmentation_line, TryFromFile(self.sequences)))
        else:
            self.seq_iter = iter(FileOrIterableReader(self.sequences))
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
