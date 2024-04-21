import typing
from bidict import bidict
import collections
import pickle
import itertools; flatten = itertools.chain.from_iterable

#try: 
from .iterator import *
#except ImportError: 
#    from segmenters import RestartableMapIterator, RestartableFlattenIterator, RestartableBatchIterator, RestartableCallableIterator

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
    
    def encode(self, seq):
        if type(seq) in [str, int]: 
            return self.encode_token(seq)
        else:
            seqiter = iter(seq)
            return [self.encode(el) for el in seqiter]

    def decode_token(self, token_idx):
        return self.word_to_idx.inv[token_idx]
    
    def decode(self, seq):
        if type(seq) in [str, int]: 
            return self.decode_token(seq)
        else:
            seqiter = iter(seq)
            return [self.decode(el) for el in seqiter]

    def build(flat_tokens:typing.Iterable=[], min_count=1, unk_token:str=default_unk_token):
        """ helper method to build a vocabulary from a stream of tokens """
        word_to_count = collections.Counter([tok for tok in flat_tokens])
        word_to_count = collections.Counter({w: c for w, c in word_to_count.items() if c >= min_count})
        word_to_idx = bidict((word, i) for i, (word, count) in enumerate(word_to_count.most_common()))
        return Vocab(word_to_idx=word_to_idx, word_to_count=word_to_count, unk_token=unk_token)
 
Key = typing.Union[str, int, typing.Set[typing.Union[str, int]]]

class Corpus:
    def __init__(self, 
                 data: typing.Iterable, 
                 segmentation: typing.Union[None, 
                                            typing.Iterable, 
                                            typing.List[typing.Iterable], 
                                            typing.Dict[str, typing.Iterable]]=None,
                 packed=True,
                 vocab: Vocab=None,
                 ) -> None:
        """Constructor

        Args:
            data (typing.Iterable): Any iterable object
            segmentation (typing.Union[None, typing.Iterable, typing.List[typing.Iterable], typing.Dict[str, typing.Iterable]], optional): Segmentations. Can be a dict, list, or a single segmentation. Defaults to None.
            packed (bool, optional): If true, indicates that in the provided segmentation format a single element is a tuple (segment_label, segment_size). Else, assumes segmentations are lists of labels where a consequtive sequence indicates a segment. Defaults to True.
            vocab (Vocab): Vocabulary object (use build_vocab function to obtain it)
        """
        self.vocab = vocab
        self.data = data
        self._packed = packed
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
                raise KeyError(f'provided key ({str(key)}) not in available segmentations ({", ".join(self.list_available_segmentations())})')
        return key
    
    def _normalize_keys(self, *keys: typing.Tuple[Key]) -> Key:
        keys = [self._normalize_key(k) for k in keys if k is not None]
        key = set.union(*keys)
        return key

    def _unpack(segmentation): 
        for label, size in segmentation: 
            for i in range(size): 
                yield label

    def _pack(segmentation): 
        for key, group in itertools.groupby(segmentation): 
            yield key, sum(1 for _ in group)

    def _default_segmentation(self): 
        data_len = sum(1 for _ in self.data)
        return map(lambda i: (i, 1), range(data_len))
    
    def _resolve_segmentation(self, *keys): 
        """ normalizes key and returns a (packed) segmentation iterator """
        if None in keys: 
            return self._default_segmentation()
        else: 
            key = self._normalize_keys(*keys)
            segmentations_single = [self.segmentations[k] for k in key]
            if self._packed: 
                # for zipping, segmentations must be unpacked
                segmentations_single = [Corpus._unpack(s) for s in segmentations_single]
            segmentation = zip(*segmentations_single) # combine segmentations by zipping
            segmentation = Corpus._pack(segmentation) # pack again
            return segmentation    

    def _resolve_segmentation_adjusted(self, coarse: Key, fine: Key): 
        """ iterates over coarse, but sizes are adjusted relative to the number of corresponding subsegments in fine (rather than elements in data) """
        seg_fine = self._resolve_segmentation(coarse, fine)
        iter_seg_fine = iter(seg_fine)
        seg_coarse = self._resolve_segmentation(coarse)
        for label_c, size_c in seg_coarse: 
            size_c_accum = 0
            num_fine_segments = 0
            while size_c_accum < size_c: 
                _, size_f = next(iter_seg_fine)
                size_c_accum += size_f
                num_fine_segments += 1
            yield label_c, num_fine_segments

    def segments(self, *segmentations: typing.Tuple[Key], as_dict=False, return_labels=False, decode=False):
        """ segmentations should be in the order of coarse -> fine """
        # normalize segmentations (each entry will be a set)
        segmentation_keys = [self._normalize_keys(key) for key in segmentations]
        # adjust the keys such that finer segmentation always contain the (boundaries of) coarser segmentations
        segmentation_keys_cumul = [set.union(*segmentation_keys[:i]) 
                                   for i in range(1, len(segmentation_keys)+1)]
        segmentation_keys_cumul.append(None)
        coarses = segmentation_keys_cumul[:-1]
        fines = segmentation_keys_cumul[1:]
        # get the (relative) segmentation iterables 
        segmentation_iterables_adj = [self._resolve_segmentation_adjusted(c, f) 
                                      for c, f in zip(coarses, fines)]
        coarsest = segmentation_iterables_adj[0]
        fines_adj = segmentation_iterables_adj[1:]
        iter_fines = [iter(f) for f in fines_adj]
        data_iter = iter(self.data)

        def consume_iters(data, label, num, *iters): 
            if len(iters) == 0: 
                _data = list(itertools.islice(data, num))
                if decode: 
                    _data = self.vocab.decode(_data)
            else: 
                _data = [consume_iters(data, *el, *iters[1:]) 
                        for el in itertools.islice(iters[0], num)]

            if as_dict: 
                return dict(data=_data, label=label)
            else: 
                if return_labels:
                    return (_data, label)
                else: 
                    return _data

        for label, size in coarsest:
            # call helper that recursively builds a dictionary/tuple containing nested segmentation
            yield consume_iters(data_iter, label, size, *iter_fines)

    def save(self, path, enumerate_iterables=True): 
        if enumerate_iterables: self._enumerate_iterables()
        with open(path, 'wb') as f: 
            pickle.dump(self, f)

    def load(path):
        with open(path, 'rb') as f: 
            return pickle.load(f)

    def build_from_lines(lines: typing.Iterable, 
                         split_line=str.split, 
                         line_index=True, 
                         min_count=1, 
                         unk_token=default_unk_token): 
        """Build corpus from lines.

        Args:
            lines (typing.Iterable): Iterable over strings that can be split by split_line.
            split_line (_type_, optional): Function that splits lines. Defaults to str.spl
            line_index (bool, optional): Whether to include a line index as a segmentation. Defaults to True.
            min_count (int, optional): Minimum word count for vocabulary building. Defaults to 1.
            unk_token (_type_, optional): Unknown token. Defaults to '<UNK>'.

        Returns:
            Corpus: Built corpus.
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
        corpus = Corpus(data=lines_split_flat_idx, 
                                 segmentation=segmentations, 
                                 packed=True, vocab=vcb)
        return corpus

    def build_conll_chunk(min_count=1, 
                          unk_token=default_unk_token, 
                          post_process=lambda x: x, 
                          char_lvl=False, 
                          vocab: Vocab=None, 
                          *args, **kwargs): 
        """Builds corpus from CoNLL formatted chunking file(s). 

        Args:
            min_count (int, optional): Minimum word count to consider when building vocabulary. Defaults to 1.
            unk_token (_type_, optional): Unknown token. Defaults to '<UNK>'.

        Returns:
            Corpus: Built corpus.
        """

        try: 
            import nltk
        except ImportError as err: 
            print("nltk is required to use build_conll_chunk (pip install nltk)")
            raise err

        reader = nltk.corpus.reader.ConllChunkCorpusReader(*args, **kwargs)
        segmentations = dict(POS=[], chunk_type=[], sent_num=[], chunk_num=[])
        if char_lvl: segmentations['word_num'] = []
        data = []
        for sent_num, sent in enumerate(reader.chunked_sents()): 
            for chunk_num, chunk in enumerate(sent): 
                if type(chunk) == tuple: 
                    chunk = [chunk]
                    chunk_type = 'punct'
                else:
                    chunk_type = chunk.label()
                for word_num, (word, POS) in enumerate(chunk): 
                    if char_lvl: 
                        for ch in word: 
                            segmentations['POS'].append(POS)
                            segmentations['chunk_type'].append(chunk_type)
                            segmentations['sent_num'].append(sent_num)
                            segmentations['chunk_num'].append(chunk_num)
                            segmentations['word_num'].append(word_num)
                            data.append(post_process(ch))
                    else: 
                        segmentations['POS'].append(POS)
                        segmentations['chunk_type'].append(chunk_type)
                        segmentations['sent_num'].append(sent_num)
                        segmentations['chunk_num'].append(chunk_num)
                        data.append(post_process(word))   
        if vocab is None: vocab = Vocab.build(flat_tokens=data, 
                                              min_count=min_count, 
                                              unk_token=unk_token)
        data_idx = RestartableMapIterator(data, vocab.encode_token)
        corpus = Corpus(data=data_idx, 
                        segmentation=segmentations, 
                        packed=False, 
                        vocab=vocab)
        return corpus
    
    def build_from_SICK(fpath, include_vocab=True):
        try: 
            import pandas as pd
        except ImportError as err: 
            print("pandas is required to use build_from_sick (pip install pandas)")
            raise err

        df = pd.read_csv(fpath, delimiter='\t', index_col='pair_ID')
        punct = ["'", ',', '-', '.', '/']

        def prep_sent(sent): 
            sent = sent.lower()
            for char in punct: 
                sent = sent.replace(char, f' {char} ')
            return [list(word) for word in sent.strip().split()]

        all_tokens = flatten(df['sentence_A'].append(df['sentence_B']).str.lower().str.replace(' ', ''))
        vocab = Vocab.build(sorted(set(all_tokens)))

        corps = {}
        for typ, A_or_B in itertools.product(['TRAIN', 'TRIAL', 'TEST'], ['A', 'B']):
            prep = [prep_sent(sent) for sent in df[df['SemEval_set']==typ][f'sentence_{A_or_B}']]
            prep_enc = vocab.encode(prep)
            seg_sent = [(i, sum(1 for word in sent for char in word)) for i, sent in enumerate(prep)]
            seg_word = [(i, len(word)) for i, word in enumerate(flatten(prep))]
            prep_enc_flat = [char for sent in prep_enc for word in sent for char in word]
            corp = Corpus(prep_enc_flat, dict(sent=seg_sent, word=seg_word), vocab=vocab, packed=True)
            corps[(typ, A_or_B)] = corp
        return corps
