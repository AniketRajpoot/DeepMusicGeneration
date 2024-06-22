import numpy as np
from enum import Enum
import torch
from vocab import *
from functools import partial
from vocab import *

SEQType = Enum('SEQType', 'Mask, Sentence, Melody, Chords, Empty, Genre')

class MusicItem():
    def __init__(self, data, vocab, ins = None, verbose=False, stream=None, position=None):
        self.data = data
        self.vocab = vocab
        #DONE
        self.ins = ins
        self._stream = stream
        self._position = position
    def __repr__(self): return '\n'.join([
        f'\n{self.__class__.__name__} - {self.data.shape}',
        f'npenc: {self.data[:10]}',
        f'{self.vocab.textify(self.data[:10])}...'])
    def __len__(self): return len(self.data)

    @classmethod
    def from_file(cls, midi_file, vocab): 
        return cls.from_stream(file2stream(midi_file), vocab)
    @classmethod
    def from_stream(cls, stream, vocab):
        if not isinstance(stream, music21.stream.Score): stream = stream.voicesToParts()
        chordarr, ins = stream2chordarr(stream) # 2.
        cls.ins = ins
        npenc = chordarr2npenc(chordarr) # 3.
        
        # print(npenc)

        #DONE
        # print('ins: ', ins)
        # print('MusicItem.from_stream: ')
        # print(chordarr.shape)
        # print(chordarr.sum())
        # print(npenc.shape)
        
        # print('npenc orig: ',npenc[:,0], npenc[:,1], npenc[:,2])
        #TODO: use `self.ins` to do correct vocab.stoi for instruments during encoding
        return cls.from_npenc(npenc, vocab, stream, ins)
    @classmethod
    def from_npenc(cls, npenc, vocab, stream=None, ins = None, genre = None): 
      # Added code for sorting the instruments in a predefined order between separation indices 
      npenc = sort_instruments(npenc, vocab)
      if(genre != None):
        return MusicItem(npenc2idxenc(npenc, vocab, ins = ins, genre = genre, seq_type=SEQType.Genre), vocab, ins = ins, stream = stream)
      else:
        return MusicItem(npenc2idxenc(npenc, vocab, ins = ins, genre = genre), vocab, ins = ins, stream = stream)

    @classmethod
    def from_idx(cls, item, vocab):
        idx,pos = item
        return MusicItem(idx, vocab=vocab, position=pos)
    def to_idx(self): return self.data, self.position

    @classmethod
    def empty(cls, vocab, seq_type=SEQType.Sentence):
        return MusicItem(seq_prefix(seq_type, vocab), vocab)

    # Added utiliy function to sort the instruments between each separation indices 
    # def sort_idx(self):
    #   self.data = sort_instruments(idxenc2npenc(self.data,self.vocab), self.vocab)
    #   return self.data

    @property
    def stream(self):
        self._stream = self.to_stream() if self._stream is None else self._stream
        return self._stream
    
    def to_stream(self, bpm=120):
        #TODO: This 'idxenc2stream' is called when MusicItem.show() is called, alter it accordingly to use self.ins to convert idx back to npenc/stream
        return idxenc2stream(self.data, self.vocab, bpm=bpm)

    def to_tensor(self, device=None):
        return to_tensor(self.data, device)
    
    def to_text(self, sep=' '): return self.vocab.textify(self.data, sep)
    
    @property
    def position(self): 
        self._position = position_enc(self.data, self.vocab) if self._position is None else self._position
        return self._position
    
    def get_pos_tensor(self, device=None): return to_tensor(self.position, device)

    def to_npenc(self):
        return idxenc2npenc(self.data, self.vocab)

    def show(self, format:str=None):
        return self.stream.show(format)
    def play(self): self.stream.show('midi')
        
    @property
    def new(self):
        return partial(type(self), vocab=self.vocab)

    def trim_to_beat(self, beat, include_last_sep=False):
        return self.new(trim_to_beat(self.data, self.position, self.vocab, beat, include_last_sep))
    
    def transpose(self, interval):
        return self.new(tfm_transpose(self.data, interval, self.vocab), position=self._position)
    
    def append(self, item):
        return self.new(np.concatenate((self.data, item.data), axis=0))
    
    def mask_pitch(self, section=None):
        return self.new(self.mask(self.vocab.note_range, section), position=self.position)
    
    def mask_duration(self, section=None, keep_position_enc=True):
        masked_data = self.mask(self.vocab.dur_range, section)
        if keep_position_enc: return self.new(masked_data, position=self.position)
        return self.new(masked_data)

    def mask(self, token_range, section_range=None):
        return mask_section(self.data, self.position, token_range, self.vocab.mask_idx, section_range=section_range)
    
    def pad_to(self, bptt):
        data = pad_seq(self.data, bptt, self.vocab.pad_idx)
        pos = pad_seq(self.position, bptt, 0)
        return self.new(data, stream=self._stream, position=pos)
    
    def split_stream_parts(self):
        self._stream = separate_melody_chord(self.stream)
        return self.stream

    def remove_eos(self):
        if self.data[-1] == self.vocab.stoi[EOS]: return self.new(self.data, stream=self.stream)
        return self

    def split_parts(self):
        return self.new(self.data, stream=separate_melody_chord(self.stream), position=self.position)
        
def pad_seq(seq, bptt, value):
    pad_len = max(bptt-seq.shape[0], 0)
    return np.pad(seq, (0, pad_len), 'constant', constant_values=value)[:bptt]

def to_tensor(t, device=None):
    t = t if isinstance(t, torch.Tensor) else torch.tensor(t)
    if device is None and torch.cuda.is_available(): t = t.cuda()
    else: t.to(device)
    return t.long()
    
def midi2idxenc(midi_file, vocab):
    "Converts midi file to index encoding for training"
    npenc = midi2npenc(midi_file) # 3.
    return npenc2idxenc(npenc, vocab)

def idxenc2stream(arr, vocab, bpm=120):
    "Converts index encoding to music21 stream"
    npenc = idxenc2npenc(arr, vocab)
    return npenc2stream(npenc, bpm=bpm)

#DONE
def npins2vocabins(x, ins:dict):
  if x in ins.keys():
    
    if(ins[x] in ACCEP_INS.keys()):
      return ACCEP_INS[ins[x]]
    else:
      return ACCEP_INS['Piano']

  elif x == (-2 - len(NOTE_TOKS) - len(DUR_TOKS)):
    return x
  else:
    raise Exception 

# single stream instead of note,dur
def npenc2idxenc(t, vocab, ins = None, genre = None, seq_type=SEQType.Sentence, add_eos=True):
    "Transforms numpy array from 2 column (note, duration) matrix to a single column"
    "[[n1, d1], [n2, d2], ...] -> [n1, d1, n2, d2]"
    if isinstance(t, (list, tuple)) and len(t) == 2: 
        return [npenc2idxenc(x, vocab, start_seq) for x in t]
    t = t.copy()
    
    #print('|npenc2idxenc|', t.shape)
    #print('chordarr ndim: ', t.ndim)
    try:
      #DONE
      if t.shape[1] == 2:
        t[:, 0] = t[:, 0] + vocab.note_range[0]
        t[:, 1] = t[:, 1] + vocab.dur_range[0]
        
        prefix = seq_prefix(seq_type, vocab)
        suffix = np.array([vocab.stoi[EOS]]) if add_eos else np.empty(0, dtype=int)   
      elif t.shape[1] == 3:
        t[:, 0] = t[:, 0] + vocab.note_range[0]
        t[:, 1] = t[:, 1] + vocab.dur_range[0]    # check wheter duration token is less than max duration 
        #SEE: `chordarr2npenc`
        #DONE
        # l1 = (t[:,2] == 5).nonzero()[0]
        # print('npenc ins: ', t[:,2][:10])

        if ins is not None:
          f = lambda x, y: npins2vocabins(x,y)
          t[:,2] = np.vectorize(f)(t[:,2], ins)
          # print('npenc ins mapped: ', t[:,2][:10])
        
        # l2 = (t[:,2] == 0).nonzero()[0]
        # print(len(l1))
        # print(len([i for i in l1 if i in l2]))

        t[:, 2] = t[:, 2] + vocab.ins_range[0]
        # print('npenc ins mapped: ',t[:,0], t[:,1], t[:,2])
        

        prefix = seq_prefix(seq_type, vocab, genre)
        suffix = np.array([vocab.stoi[EOS]]) if add_eos else np.empty(0, dtype=int)
        # print('npenc2idxenc result: ',np.concatenate([t.reshape(-1), suffix]))
    except IndexError as e:
        print('IndexError, t.shape:', t.shape )
        raise e
    return np.concatenate([prefix, t.reshape(-1), suffix])

def seq_prefix(seq_type, vocab, genre = None):
    if seq_type == SEQType.Empty: return np.empty(0, dtype=int)
    start_token = vocab.bos_idx
    if seq_type == SEQType.Chords: start_token = vocab.stoi[CSEQ]
    if seq_type == SEQType.Melody: start_token = vocab.stoi[MSEQ]
    if seq_type == SEQType.Genre and genre != None:
      token = BOS
      genre = genre.lower() 
      if('electronic' in genre): token = ELECTRONIC  
      elif('folk' in genre): token = FOLK
      elif('funk' in genre): token = FUNK
      elif('jazz' in genre): token = JAZZ
      elif('pop' in genre): token = POP
      elif('rock' in genre): token = ROCK
      start_token = vocab.stoi[token]
    return np.array([start_token, vocab.pad_idx])

#IMP
#TODO: Use MusicItem/self.ins to verify proper reconversion into npenc of shape [X, 3]
def idxenc2npenc(t, vocab, validate=True):
    if validate: 
      t = to_valid_idxenc(t, vocab.npenc_range)
      
    #DONE
    # t = t.copy().reshape(-1, 2)
    # if t.shape[-1]%3 != 0 : 
    #   t = t[:(t.shape[-1] - t.shape[-1]%3)]

    # temp = vocab.textify(t, ' ').split(' ')
    # print(temp[:50])
    # print([x for index,x in enumerate(temp) if index%3 == 0])

    # print('idxenc2npenc : ')
    # print('t = ',  t)
    ins_toks = [True if vocab.is_ins(x) else False for x in t]
    # print(ins_toks)
    last_idx_tok_idx_rev = ins_toks[::-1].index(True)
    # print(last_idx_tok_idx_rev)
    # print('Last elements before proc: ', t[-5:])
    t = t[:(len(ins_toks) - last_idx_tok_idx_rev)]
    # print('Last elements after proc: ', t[-5:])


    t = t.copy().reshape(-1, 3)

    # print('idxenc2npenc input: ', t[:,0], t[:,1], t[:,2]) 

    

    if t.shape[0] == 0: return t
        
    t[:, 0] = t[:, 0] - vocab.note_range[0]
    t[:, 1] = t[:, 1] - vocab.dur_range[0]
    t[:, 2] = t[:, 2] - vocab.ins_range[0]

    # print('idxenc2npenc: ', t[:,0], t[:,1], t[:,2] )

    #TODO: generalise to `ndi`, currently for `nd`
    if validate:
      t = to_valid_npenc(t)
    return t

def to_valid_idxenc(t, valid_range):
    r = valid_range
    t = t[np.where((t >= r[0]) & (t < r[1]))]
    #DONE
    #Removed this line that removes the odd dimension, which is required by our nxdxi array of shape [X, 3]
    #if t.shape[-1] % 2 == 1: t = t[..., :-1]
    return t

def to_valid_npenc(t):
    is_note = (t[:, 0] < VALTSEP) | (t[:, 0] >= NOTE_SIZE)
    invalid_note_idx = is_note.argmax()
    invalid_dur_idx = (t[:, 1] < 0).argmax()

    invalid_idx = max(invalid_dur_idx, invalid_note_idx)
    if invalid_idx > 0: 
        if invalid_note_idx > 0 and invalid_dur_idx > 0: invalid_idx = min(invalid_dur_idx, invalid_note_idx)
        print('Non midi note detected. Only returning valid portion. Index, seed', invalid_idx, t.shape)
        return t[:invalid_idx]
    return t

def sort_instruments(npenc, vocab):
    "Sorts instrument according to accept instrument list"
    sep_idxs = (npenc[:,0] == -1).nonzero()[0]
    
    updated_npenc = []

    first_sep = sep_idxs[0]
    
    if(first_sep != 0):
      npenc_sub = npenc[0 : first_sep]
      npenc_sub = sorted(npenc_sub, key = lambda x : x[2])
      final_subset = npenc_sub
      updated_npenc.extend(final_subset)

    for e in zip(sep_idxs[:-1],sep_idxs[1:]):
      npenc_sub = npenc[e[0] + 1 : e[1]]
      npenc_sub = sorted(npenc_sub, key = lambda x : x[2])
      # npenc_sub = [list(i) for i in npenc_sub]
      sep = npenc[e[0]]
      final_subset = [sep] + npenc_sub

      # print(final_subset)
      updated_npenc.extend(final_subset)
    
    last_sep = sep_idxs[-1]
    
    if(len(npenc) > last_sep + 1):
      npenc_sub = npenc[last_sep + 1 : ]
      npenc_sub = sorted(npenc_sub, key = lambda x : x[2])
      sep = npenc[e[0]]
      final_subset = [sep] + npenc_sub
    else:
      final_subset = [sep]

    updated_npenc.extend(final_subset)

    updated_npenc = np.array(updated_npenc)
    sep_idxs_updated = (updated_npenc[:,0] == -1).nonzero()[0]

    # print(sep_idxs)
    # print(sep_idxs_updated)

    assert list(sep_idxs) == list(sep_idxs_updated)
    # print(updated_npenc)
    return updated_npenc

def position_enc(idxenc, vocab):
    "Calculates positional beat encoding."
    
    # print('position encoding : \n')
    # print(idxenc)

    # gets the separation tokens indices 
    sep_idxs = (idxenc == vocab.sep_idx).nonzero()[0]

    
    sep_idxs = sep_idxs[sep_idxs+2 < idxenc.shape[0]] # remove any indexes right before out of bounds (sep_idx+2)
    dur_vals = idxenc[sep_idxs+1]

    dur_vals[dur_vals == vocab.mask_idx] = vocab.dur_range[0] # make sure masked durations are 0
    dur_vals -= vocab.dur_range[0]
    
    posenc = np.zeros_like(idxenc)

    # DONE : changed to account for xxni token  
    # posenc[sep_idxs+2] = dur_vals

    
    # print(sep_idxs)
    # print(sep_idxs[:-1])

    # last_sep_idx = sep_idxs[-1]
    # last_dur_val = dur_vals[-1]
    try:
      if(len(idxenc) > sep_idxs[-1] + 3):
        posenc[sep_idxs+3] = dur_vals
      else:
        sep_idxs = sep_idxs[:-1]
        dur_vals = dur_vals[:-1]
        posenc[sep_idxs+3] = dur_vals
    except:
      print('idx_enc = ', idxenc)
      print('sep_idxs = ', sep_idxs )

    return posenc.cumsum()

def beat2index(idxenc, pos, vocab, beat, include_last_sep=False):
    cutoff = find_beat(pos, beat)
    if cutoff < 2: return 2 # always leave starter tokens
    if len(idxenc) < 2 or include_last_sep: return cutoff
    if idxenc[cutoff - 2] == vocab.sep_idx: return cutoff - 2
    return cutoff

def find_beat(pos, beat, sample_freq=SAMPLE_FREQ, side='left'):
    return np.searchsorted(pos, beat * sample_freq, side=side)

# TRANSFORMS

def tfm_transpose(x, value, vocab):
    x = x.copy()
    x[(x >= vocab.note_range[0]) & (x < vocab.note_range[1])] += value
    return x

def trim_to_beat(idxenc, pos, vocab, to_beat=None, include_last_sep=True):
    if to_beat is None: return idxenc
    cutoff = beat2index(idxenc, pos, vocab, to_beat, include_last_sep=include_last_sep)
    return idxenc[:cutoff]

def mask_input(xb, mask_range, replacement_idx):
    xb = xb.copy()
    xb[(xb >= mask_range[0]) & (xb < mask_range[1])] = replacement_idx
    return xb

def mask_section(xb, pos, token_range, replacement_idx, section_range=None):
    xb = xb.copy()
    token_mask = (xb >= token_range[0]) & (xb < token_range[1])

    if section_range is None: section_range = (None, None)
    section_mask = np.zeros_like(xb, dtype=bool)
    start_idx = find_beat(pos, section_range[0]) if section_range[0] is not None else 0
    end_idx = find_beat(pos, section_range[1]) if section_range[1] is not None else xb.shape[0]
    section_mask[start_idx:end_idx] = True
    
    xb[token_mask & section_mask] = replacement_idx
    return xb