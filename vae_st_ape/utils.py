import random
from collections import UserList, defaultdict
import numpy as np
import pandas as pd
import torch

class SpecialTokens:
    bos = '<bos>'
    eos = '<eos>'
    pad = '<pad>'
    unk = '<unk>'

class CharVocab:
    @classmethod
    def from_data(cls, data, *args, **kwargs):
        chars = set()
        for string in data:
            chars.update(string)
        return cls(chars, *args, **kwargs)

    def __init__(self, chars, ss=SpecialTokens):
        if (ss.bos in chars) or (ss.eos in chars) or (ss.pad in chars) or (ss.unk in chars):
            raise ValueError('SpecialTokens in chars')
        all_syms = sorted(list(chars)) + [ss.bos, ss.eos, ss.pad, ss.unk]
        self.ss = ss
        self.c2i = {c: i for i, c in enumerate(all_syms)}
        self.i2c = {i: c for i, c in enumerate(all_syms)}
    def __len__(self):
        return len(self.c2i)
    @property
    def bos(self):
        return self.c2i[self.ss.bos]
    @property
    def eos(self):
        return self.c2i[self.ss.eos]
    @property
    def pad(self):
        return self.c2i[self.ss.pad]
    @property
    def unk(self):
        return self.c2i[self.ss.unk]
    def char2id(self, char):
        if char not in self.c2i:
            return self.unk
        return self.c2i[char]
    def id2char(self, id):
        if id not in self.i2c:
            return self.ss.unk
        return self.i2c[id]
    def string2ids(self, string, add_bos=False, add_eos=False):
        ids = [self.char2id(c) for c in string]
        if add_bos:
            ids = [self.bos] + ids
        if add_eos:
            ids = ids + [self.eos]
        return ids
    def ids2string(self, ids, rem_bos=True, rem_eos=True):
        if len(ids) == 0:
            return ''
        if rem_bos and ids[0] == self.bos:
            ids = ids[1:]
        if rem_eos and ids[-1] == self.eos:
            ids = ids[:-1]
        string = ''.join([self.id2char(id) for id in ids])
        return string

class OneHotVocab(CharVocab):
    def __init__(self, *args, **kwargs):
        super(OneHotVocab, self).__init__(*args, **kwargs)
        self.vectors = torch.eye(len(self.c2i))

class Logger(UserList):
    def __init__(self, data=None):
        super().__init__()
        self.sdata = defaultdict(list)
        for step in (data or []):
            self.append(step)
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        if isinstance(key, slice):
            return Logger(self.data[key])
        ldata = self.sdata[key]
        if isinstance(ldata[0], dict):
            return Logger(ldata)
        return ldata
    def append(self, step_dict):
        super().append(step_dict)
        for k, v in step_dict.items():
            self.sdata[k].append(v)
    def save(self, path):
        df = pd.DataFrame(list(self))
        df.to_csv(path, index=None)

class CircularBuffer:
    def __init__(self, size):
        self.max_size = size
        self.data = np.zeros(self.max_size)
        self.size = 0
        self.pointer = -1
    def add(self, element):
        self.size = min(self.size + 1, self.max_size)
        self.pointer = (self.pointer + 1) % self.max_size
        self.data[self.pointer] = element
        return element
    def last(self):
        assert self.pointer != -1, "Can't get an element from an empty buffer!"
        return self.data[self.pointer]
    def mean(self):
        if self.size > 0:
            return self.data[:self.size].mean()
        return 0.0
