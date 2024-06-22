from typing import Collection, List
from fastai.basics import *
from encodings import *
# from .music_transformer import transform
from functools import partial
import pickle

class MusicVocab():
    "Contain the correspondence between numbers and tokens and numericalize."
    def __init__(self, itos:Collection[str]):
        self.itos = itos
        self.stoi = {v:k for k,v in enumerate(self.itos)}

    def numericalize(self, t:Collection[str]) -> List[int]:
        "Convert a list of tokens `t` to their ids."
        return [self.stoi[w] for w in t]

    def textify(self, nums:Collection[int], sep=' ') -> List[str]:
        "Convert a list of `nums` to their tokens."
        items = [self.itos[i] for i in nums]
        return sep.join(items) if sep is not None else items
    
    #DONE
    def to_music_item(self, idxenc, ins = None):
        return MusicItem(idxenc, self, ins)
    
    @property 
    def mask_idx(self): return self.stoi[MASK]
    @property 
    def pad_idx(self): return self.stoi[PAD]
    @property
    def bos_idx(self): return self.stoi[BOS]
    @property
    def sep_idx(self): return self.stoi[SEP]
    #DONE
    @property
    def ni_idx(self): return self.stoi[IN]
    @property
    #DONE: changed 'DUR_END' to 'INS_END'
    def npenc_range(self): return (self.stoi[IN], self.stoi[INS_END]+1)
    @property
    def note_range(self): return self.stoi[NOTE_START], self.stoi[NOTE_END]+1
    @property
    def dur_range(self): return self.stoi[DUR_START], self.stoi[DUR_END]+1
    #DONE
    @property
    def ins_range(self): return self.stoi[INS_START], self.stoi[INS_END]+1

    def is_duration(self, idx): 
        return idx >= self.dur_range[0] and idx < self.dur_range[1]
    def is_duration_or_pad(self, idx):
        return idx == self.pad_idx or self.is_duration(idx)
    #DONE
    def is_note(self, idx): 
        return idx == self.sep_idx or (idx >= self.note_range[0] and idx < self.note_range[1])
    def is_ins(self, idx):
        return idx == self.ni_idx or (idx >= self.ins_range[0] and idx < self.ins_range[1])
    def __getstate__(self):
        return {'itos':self.itos}

    def __setstate__(self, state:dict):
        self.itos = state['itos']
        self.stoi = {v:k for k,v in enumerate(self.itos)}
        
    def __len__(self): return len(self.itos)

    def save(self, path):
        "Save `self.itos` in `path`"
        pickle.dump(self.itos, open(path, 'wb'))

    @classmethod
    def create(cls) -> 'Vocab':
        "Create a vocabulary from a set of `tokens`."
        #DONE
        #itos = SPECIAL_TOKS + NOTE_TOKS + DUR_TOKS + MTEMPO_TOKS
        itos = SPECIAL_TOKS + NOTE_TOKS + DUR_TOKS + INS_TOKS + MTEMPO_TOKS
        
        if len(itos)%8 != 0:
            itos = itos + [f'dummy{i}' for i in range(len(itos)%8)]
        return cls(itos)
    
    @classmethod
    def load(cls, path):
        "Load the `Vocab` contained in `path`"
        itos = pickle.load(open(path, 'rb'))
        return cls(itos)