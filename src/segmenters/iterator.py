from collections import deque

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

def single_subiter(x):
    yield x

class FileReader:
    def __init__(self, fpath, subiter_fn=single_subiter):
        self.fpath=fpath
        self.subiter_fn=subiter_fn
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

def word2subwords(word):
    return word.split('|')[0].split('-')

class ByLineReader(FileReader):
    # in file of 'A-B-C D-E-F\\nG-H-I' yields [A,B,C,D,E,F]
    def __init__(self, fpath, max_context_size=10, step_size=5):
        def subiter_fn(line):
            tokens = [subword for word in line.split() for subword in word2subwords(word)]
            # use sliding window to constrain maximum context size
            for subseq in SlidingWindow(tokens, max_context_size, step_size):
                yield subseq
        super().__init__(fpath, subiter_fn)

class ByWordReader(FileReader):
    # in file of 'A-B-C D-E-F\\nG-H-I' yields [A,B,C] 
    def __init__(self, fpath):
        def subiter_fn(line):
            for word in line.split():
                yield word2subwords(word)
        super().__init__(fpath, subiter_fn)

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
                self.deque.popleft()
        # fill up the deque
        while len(self.deque) < self.win_size + 1:
            try:
                self.deque.append(next(self.iter))
            except StopIteration:
                self.stopnext = True
                break
                
        return list(self.deque)[:self.win_size]
#print(list(sliding_window(range(5), 2, 2)))
