import typing
from bidict import bidict
import collections
import nltk
import pickle
import itertools
import nltk

try: 
    from .iterator import *
except ImportError: 
    from segmenters import RestartableMapIterator, RestartableFlattenIterator, RestartableBatchIterator, RestartableCallableIterator

default_unk_token = '<UNK>'

class Vocab:
    word_to_idx: bidict
    word_to_count: dict
    
    def __init__(self, word_to_idx: bidict, word_to_count: dict, unk_token=default_unk_token):
        self.word_to_count=word_to_count
        self.word_to_idx=word_to_idx
        self.unk_token=unk_token
        try: 
            self.add_type(unk_token)
        except ValueError: 
            pass

    def __str__(self): 
        return str(self.word_to_idx)

    def __len__(self):
        return len(self.word_to_idx)

    def _get_new_word_id(self): 
        return max(self.word_to_idx.values())+1

    def add_type(self, token, count=0):
        if token in self.word_to_idx: 
            raise ValueError(f"token '{token}' is already in the vocabulary")
        self.word_to_idx[token] = self._get_new_word_id()
        self.word_to_count[token] = count
        return token

    def token_count(self):
        return int(sum(self.word_to_count.values()))

    def encode_token(self, token):
        return self.word_to_idx.get(token, self.word_to_idx[self.unk_token])
    
    def encode_sent(self, sent):
        return [self.encode_token(token) for token in sent]

    def encode_batch(self, sents):
        return [self.encode_sent(sent) for sent in sents]

    def decode_token(self, token_idx):
        return self.word_to_idx.inv[token_idx]

    def decode_sent(self, sent, stringify=False):
        tokens = [self.decode_token(token) for token in sent]
        if stringify: tokens = ' '.join(tokens)
        return tokens

    def decode_batch(self, sents, stringify=False):
        return [self.decode_sent(sent, stringify) for sent in sents]
    
    def build(flat_tokens:typing.Iterable=[], min_count=1, unk_token:str=default_unk_token):
        """ helper method to build a vocabulary from a stream of tokens """
        word_to_count = collections.Counter([tok for tok in flat_tokens])
        word_to_count = collections.Counter({w: c for w, c in word_to_count.items() if c >= min_count})
        word_to_idx = bidict((word, i) for i, (word, count) in enumerate(word_to_count.most_common()))
        return Vocab(word_to_idx=word_to_idx, word_to_count=word_to_count, unk_token=unk_token)
    
#vcb = Vocab.build('salkjdfhaszlkjdfhaskljdfhaslkjdfhajksdfhadsjklfhaslkjdfhasdkljfhasdlfj', min_count=1)
#vcb.word_to_idx, vcb.word_to_count, vcb.encode_sent('blabla')
    
Key = typing.Union[str, int, typing.Set[typing.Union[str, int]]]

class SegmentedCorpus:
    def __init__(self, data: typing.Iterable, 
                 segmentation: typing.Union[None, typing.Iterable, typing.List[typing.Iterable], typing.Dict[str, typing.Iterable]]=None,
                 packed=True,
                 vocab: Vocab=None) -> None:
        """Constructor

        Args:
            vocab (Vocab): Vocabulary object (use build_vocab function to obtain it)
            data (typing.Iterable): Any iterable object
            segmentation (typing.Union[None, typing.Iterable, typing.List[typing.Iterable], typing.Dict[str, typing.Iterable]], optional): Segmentations. Can be a dict, list, or a single segmentation. Defaults to None.
            packed (bool, optional): If true, indicates that in the provided segmentation format a single element is a tuple (segment_label, segment_size). Else, assumes segmentations are lists of labels where a consequtive sequence indicates a segment. Defaults to True.
        """
        self.vocab = vocab
        self.data = data
        self.packed = packed
        if type(segmentation) == list: 
            self.segmentations = {i: s for i, s in enumerate(segmentation)}
        elif type(segmentation) == dict: 
            self.segmentations = segmentation
        else: 
            self.segmentations = {0: segmentation}

    def list_available_segmentations(self):
        return list(self.segmentations.keys())
    
    def _enumerate_iterables(self): 
        if type(self.data) != list:
            self.data = list(self.data)
        for key in self.segmentations.keys(): 
            if type(self.segmentations[key]) != list: 
                self.segmentations[key] = list(self.segmentations[key])
    
    def _normalize_key(self, key: Key):
        if key is None: 
            return key
        if type(key) == str or type(key) == int: 
            key = set([key])
        key = set([_ for _ in key if _ is not None])
        for single in key: 
            if single not in self.segmentations: 
                raise KeyError(f'provided key (single) not in available segmentations ({",".join(self.list_available_segmentations())})')
        return key
    
    def _default_segmentation(self): 
        data_len = sum(1 for _ in self.data)
        return map(lambda i: (i, 1), range(data_len))
    
    def _resolve_segmentation(self, *keys): 
        """ normalizes key and returns a (packed) segmentation iterator """
        if keys[0] is None: 
            return self._default_segmentation()
        else: 
            keys = [self._normalize_key(k) for k in keys if k is not None]
            key = set.union(*keys)
            segmentations_single = [self.segmentations[k] for k in key]
            if self.packed: 
                # for zipping, segmentations must be unpacked
                segmentations_single = [SegmentedCorpus._unpack(s) for s in segmentations_single]
            segmentation = zip(*segmentations_single) # combine segmentations by zipping
            segmentation = SegmentedCorpus._pack(segmentation) # pack again
            return segmentation
    
    def _unpack(segmentation): 
        for label, size in segmentation: 
            for i in range(size): 
                yield label

    def _pack(segmentation): 
        for key, group in itertools.groupby(segmentation): 
            yield key, sum(1 for _ in group)

    def segments(self, coarse: Key, fine: Key=None):
        seg_coarse = self._resolve_segmentation(coarse)
        iter_data = iter(self.data)

        if fine is None: 
            for label, size in seg_coarse:
                _data = [next(iter_data) for i in range(size)]
                segment = {'data': _data, 'label': label}
                yield segment
        else: 
            seg_fine = self._resolve_segmentation(fine, coarse)
            iter_seg_fine = iter(seg_fine)
            iter_seg_coarse = iter(seg_coarse)
            while True: 
                try: key_coarse, size_coarse = next(iter_seg_coarse)
                except StopIteration: break
                segment = []
                while size_coarse > 0: 
                    key_fine, size_fine = next(iter_seg_fine)
                    segment.append({
                        'data': [next(iter_data) for i in range(size_fine)], 
                        'label_fine': key_fine})
                    size_coarse -= size_fine
                yield {'segments': segment, 'label_coarse': key_coarse}        

    def save(self, path, enumerate_iterables=True): 
        if enumerate_iterables: self._enumerate_iterables()
        with open(path, 'wb') as f: 
            pickle.dump(self, f)

    def load(path):
        with open(path, 'rb') as f: 
            return pickle.load(f)

    def build_from_lines(lines: typing.Iterable, split_line=str.split, line_index=True, min_count=1, unk_token=default_unk_token): 
        """Build corpus from lines.

        Args:
            lines (typing.Iterable): Iterable over strings that can be split by split_line.
            split_line (_type_, optional): Function that splits lines. Defaults to str.spl
            line_index (bool, optional): Whether to include a line index as a segmentation. Defaults to True.
            min_count (int, optional): Minimum word count for vocabulary building. Defaults to 1.
            unk_token (_type_, optional): Unknown token. Defaults to '<UNK>'.

        Returns:
            SegmentedCorpus: Built corpus.
        """
        lines_split = RestartableMapIterator(lines, split_line)
        lines_split_flat = RestartableFlattenIterator(lines_split)
        vcb = Vocab.build(flat_tokens=lines_split_flat, 
                          min_count=min_count, unk_token=unk_token)
        lines_split_flat_idx = RestartableMapIterator(
            lines_split_flat, vcb.encode_token)
        segmentations={}
        if line_index: 
            segmentations['line_num'] = []
            for i, line in enumerate(lines_split):
                segmentations['line_num'].append((i, len(line)))
        corpus = SegmentedCorpus(data=lines_split_flat_idx, 
                                 segmentation=segmentations, 
                                 packed=True, vocab=vcb)
        return corpus

    def build_conll(min_count=1, unk_token=default_unk_token, *args, **kwargs): 
        """Builds corpus from CoNLL formatted file(s). 

        Args:
            min_count (int, optional): Minimum word count to consider when building vocabulary. Defaults to 1.
            unk_token (_type_, optional): Unknown token. Defaults to '<UNK>'.

        Returns:
            SegmentedCorpus: Built corpus.
        """
        reader = nltk.corpus.reader.ConllChunkCorpusReader(*args, **kwargs)
        segmentations = dict(POS=[], chunk_type=[], sent_num=[], chunk_num=[])
        data = []
        for i, sent in enumerate(reader.chunked_sents()): 
            for j, chunk in enumerate(sent): 
                if type(chunk) == tuple: 
                    chunk = [chunk]
                    chunk_type = 'punct'
                else:
                    chunk_type = chunk.label()
                for word, POS in chunk: 
                    segmentations['POS'].append(POS)
                    segmentations['chunk_type'].append(chunk_type)
                    segmentations['sent_num'].append(i)
                    segmentations['chunk_num'].append(j)
                    data.append(word)
        vcb = Vocab.build(flat_tokens=data, min_count=min_count, 
                          unk_token=unk_token)
        data_idx = RestartableMapIterator(data, vcb.encode_token)
        corpus = SegmentedCorpus(data=data_idx, 
                                 segmentation=segmentations, 
                                 packed=False, vocab=vcb)
        return corpus
    
if __name__ == '__main__': 
    corpus = SegmentedCorpus.build_conll(root='../corpora/conll2000/', fileids=['test.txt'], chunk_types=None)
    print(corpus.list_available_segmentations())
    corpus.save('connl.pkl')
    corpus = SegmentedCorpus.load('connl.pkl')
    for segments in itertools.islice(corpus.segments(('chunk_type', 'chunk_num'), 'POS'), 3): 
        print(segments['label_coarse'])
        for segment in segments['segments']:
            print("\t", segment['label_fine'], corpus.vocab.decode_sent(segment['data']))
    for Seg in itertools.islice(corpus.segments(coarse='sent_num', fine='chunk_num'), 5): 
        print(f'sent. {Seg["label_coarse"]}')
        for seg in Seg['segments']: 
            print(f'chunk {seg["label_fine"]}')
            print(corpus.vocab.decode_sent(seg['data']))
            
    corpus = SegmentedCorpus.build_from_lines([
        'hello there', 
        'how are you ?',
    ], split_line=str.split, min_count=1, unk_token='<UNK>')

    for line in corpus.segments(None):
        print(corpus.vocab.decode_sent(line['data']), line['label'])
    for line in corpus.segments('line_num'):
        print(corpus.vocab.decode_sent(line['data']), line['label'])
    print()

    s0 = [1,1,1,1,1,1,1,1,1]
    s1 = [1,1,1,1,2,2,3,3,3]
    s2 = [1,1,2,3,4,4,4,4,5]
    seq = range(len(s1))
    vcb = Vocab.build(seq)
    sc = SegmentedCorpus(seq, [s0, s1, s2], False, vcb)
    for seg in sc.segments((0, 1), 2): 
        print(seg)
    for seg in sc.segments(0): 
        print(seg)

    s0 = [(1,9)]
    s1 = [(1,4), (2,2), (3,3)]
    s2 = [(1,2), (2,1), (3,1), (4,4), (5,1)]
    seq = range(9)
    vcb = Vocab.build(seq)
    sc = SegmentedCorpus(seq, [s0, s1, s2], True, vcb)
    print()
    for seg in sc.segments((0, 1), 2): 
        print(seg)
    for seg in sc.segments(0): 
        print(seg)