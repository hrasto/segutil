from collections import deque
import os

class RestartableMapIterator:
    def __init__(self, iterable, fn):
        self.iterable = iterable
        self.fn = fn
    def __iter__(self):
        self.iter = iter(map(self.fn, self.iterable))
        return self
    def __next__(self):
        return next(self.iter)

def batchify(sequence, batch_size, batch_overlap=0, batch_size_min=0):
    if not batch_overlap < batch_size: 
        raise Exception(f"batch_overlap {batch_overlap} must be smaller than batch_size {batch_size}")
    iterator = iter(SlidingWindow(sequence, batch_size, batch_size-batch_overlap))
    for batch in iterator:
        if len(batch) > batch_size_min:
            yield batch

class RestartableBatchIterator:
    def __init__(self, iterable, batch_size):
        self.iterable = iterable
        self.batch_size = batch_size
    def __iter__(self):
        self.iter = iter(batchify(self.iterable, self.batch_size))
        return self
    def __next__(self):
        return next(self.iter)

class FileReader:
    def __init__(self, fpath):
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f'no file found at location {fpath}')
        self.fpath=fpath
    def __iter__(self):
        self.file=open(self.fpath, 'r')
        self.file_iter=iter(self.file)
        self.line_iter=None
        return self
    def __next__(self):
        if self.line_iter is not None:
            try: return next(self.line_iter)
            except StopIteration: pass
        try: 
            nextline = next(self.file_iter)
            self.line_iter = iter(self.subiter_fn(nextline))
            # if code reaches here, self.sub_iter is an iterator
            return self.__next__()
        except StopIteration: 
            self.file_iter.close()
            raise StopIteration()
    def subiter_fn(self, x):
        return x

def line2subwords(line):
    return [subword for word in line.split() for subword in word.split('|')[0].split('-')]

def line2characters(line):
    return [ch for ch in line.strip() if ch not in ['|', '-']]

def line2words(line):
    return [word.split('|')[0].replace('-', '') for word in line.split()]

class LineContextReader(FileReader):
    # in file of 'A-B-C D-E-F\\nG-H-I' yields [A,B,C,D,E,F]
    def __init__(self, fpath, line2tokens, max_num_words=10, strict_cutoff=True):
        #TODO consider adding a 'min_context_size' parameter
        self.tokenize = line2tokens
        self.max_num_words = max_num_words
        self.strict_cutoff = strict_cutoff
        super().__init__(fpath)

    def subiter_fn(self, line):
        words = line.split()
        # use sliding window to constrain maximum context size
        for subseq in SlidingWindow(words, self.max_num_words, 99999 if self.strict_cutoff else self.max_num_words):
            subseq = ' '.join(subseq)
            tokens = self.tokenize(subseq)
            yield tokens

class WordContextReader(FileReader):
    # in file of 'A-B-C D-E-F\\nG-H-I' yields [A,B,C] 
    def __init__(self, fpath, line2tokens):
        self.tokenize = line2tokens
        super().__init__(fpath)

    def subiter_fn(self, line):
        for word in line.split():
            tokens = self.line2tokens(word)
            yield tokens

#corpus_reader = ByLineReader('/Users/rastislavhronsky/ml-experiments/experiments-spring-2023/corpus/toy/PC_0dot00_CP_0dot00_MI_0dot00/corpus.txt')
#for word in corpus_reader:
#    print(word)
#exit()

"""
def read_corpus_by_line(fpath, max_context_size=10, step_size=5):
    # in file of 'A-B-C D-E-F\\nG-H-I' yields [A,B,C,D,E,F]
    with open(fpath, 'r') as f: 
        for line in f:
            tokens = [subword for word in line.split() for subword in word.split('-')]
            # use sliding window to constrain maximum context size
            for subseq in sliding_window(tokens, max_context_size, step_size):
                yield subseq

def read_corpus_by_word(fpath):
    # in file of 'A-B-C D-E-F\\nG-H-I' yields [A,B,C] 
    with open(fpath, 'r') as f: 
        for line in f:
            for word in line.split():
                yield word.split('-')
"""

class MaskedIterator:
    def __init__(self, iterable, mask):
        self.mask=mask
        self.iterable=iterable
    def __iter__(self):
        self.iter = iter(self.iterable)
        self.iter_mask = iter(self.mask)
        return self
    def __next__(self):
        keep, item = False, None
        while not keep: 
            keep = next(self.iter_mask)
            item = next(self.iter)
        return item

class SlidingWindow:
    def __init__(self, iterable, win_size, step=1):
        if step < 1:
            raise Exception("step must be > 1")
        self.iterable = iterable
        self.win_size = win_size
        self.step = step

    def __iter__(self):
        self.iter = iter(self.iterable)
        self.deque = deque()
        self.stopnext = False
        return self

    def __next__(self):
        if self.stopnext:
            raise StopIteration()

        # remove old elements
        if len(self.deque) > 0:
            for _ in range(self.step):
                try: 
                    self.deque.popleft()
                except IndexError: 
                    raise StopIteration()
        # fill up the deque
        while len(self.deque) < self.win_size + 1:
            try:
                self.deque.append(next(self.iter))
            except StopIteration:
                self.stopnext = True
                break
                
        return list(self.deque)[:self.win_size]
#print(list(sliding_window(range(5), 2, 2)))
