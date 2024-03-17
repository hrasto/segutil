import typing
from bidict import bidict
import collections
import nltk
import pickle
import itertools

try: 
    from . import iterator as it
except ImportError: 
    import segmenters.iterator as it

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
    
def build_vocab(flat_tokens:typing.Iterable=[], min_count=1, unk_token:str=default_unk_token):
    """ helper method to build a vocabulary from a stream of tokens """
    word_to_count = collections.Counter([tok for tok in flat_tokens])
    word_to_count = collections.Counter({w: c for w, c in word_to_count.items() if c >= min_count})
    word_to_idx = bidict((word, i) for i, (word, count) in enumerate(word_to_count.most_common()))
    return Vocab(word_to_idx=word_to_idx, word_to_count=word_to_count, unk_token=unk_token)

Key = typing.Union[str, int, typing.Set[typing.Union[str, int]]]

class SegmentedCorpus:
    def __init__(self, vocab: Vocab, data: typing.Iterable, 
                 segmentation: typing.Union[None, typing.Iterable, typing.List[typing.Iterable], typing.Dict[str, typing.Iterable]]=None,
                 packed=True) -> None:
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

    def segments(self, key: Key):
        segmentation = self._resolve_segmentation(key)
        iter_data = iter(self.data)
        for label, size in segmentation:
            _data = [next(iter_data) for i in range(size)]
            segment = {'data': _data, 'label': label}
            yield segment

    def segments_wrt(self, coarse:Key, fine:Key):
        iter_data = iter(self.data)
        seg_fine = self._resolve_segmentation(fine, coarse)
        seg_coarse = self._resolve_segmentation(coarse)
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

    def build_from_lines(lines: typing.Iterable, split_line=str.split, line_index=True, segmentation_name='line_num', **kwargs): 
        lines_split = it.RestartableMapIterator(lines, split_line)
        lines_split_flat = it.RestartableFlattenIterator(lines_split)
        vcb = build_vocab(lines_split_flat, **kwargs)
        
        segmentations={}
        if line_index: 
            segmentations[segmentation_name] = []
            for i, line in enumerate(lines_split):
                segmentations[segmentation_name].append((i, len(line)))

        corpus = SegmentedCorpus(vcb, lines_split_flat, segmentations, True)
        return corpus

    def build_conll(): 
        pass