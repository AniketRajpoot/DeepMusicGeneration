## **Imports**

#@title
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
import music21
import os

import numpy as np
# from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from enum import Enum
import matplotlib.pyplot as plt
from typing import *
import math
import time
import pickle
import shutil

from fastai.callbacks import EarlyStoppingCallback, SaveModelCallback, ReduceLROnPlateauCallback

## **Utils**

### **Declaration of Helper Variables**

#@title
from enum import Enum
import music21

PIANO_TYPES = list(range(24)) + list(range(80, 96)) # Piano, Synths
PLUCK_TYPES = list(range(24, 40)) + list(range(104, 112)) # Guitar, Bass, Ethnic
BRIGHT_TYPES = list(range(40, 56)) + list(range(56, 80))

PIANO_RANGE = (21, 109) # https://en.wikipedia.org/wiki/Scientific_pitch_notation


#@title
#Using enums in python
class Track(Enum):
    PIANO = 0 # discrete instruments - keyboard, woodwinds
    PLUCK = 1 # continuous instruments with pitch bend: violin, trombone, synths
    BRIGHT = 2
    PERC = 3
    UNDEF = 4
    
ype2inst = {
    # use print_music21_instruments() to see supported types
    Track.PIANO: 0, # Piano
    Track.PLUCK: 24, # Guitar
    Track.BRIGHT: 40, # Violin
    Track.PERC: 114, # Steel Drum
}

# INFO_TYPES = set(['TIME_SIGNATURE', 'KEY_SIGNATURE'])
INFO_TYPES = set(['TIME_SIGNATURE', 'KEY_SIGNATURE', 'SET_TEMPO'])


#@title
def file2mf(fp):
    mf = music21.midi.MidiFile()
    if isinstance(fp, bytes):
        mf.readstr(fp)
    else:
        mf.open(fp)
        mf.read()
        mf.close()
    return mf

def mf2stream(mf): return music21.midi.translate.midiFileToStream(mf)

def is_empty_midi(fp):
    if fp is None: return False
    mf = file2mf(fp)
    return not any([t.hasNotes() for t in mf.tracks])

def num_piano_tracks(fp):
    music_file = file2mf(fp)
    note_tracks = [t for t in music_file.tracks if t.hasNotes() and get_track_type(t) == Track.PIANO]
    return len(note_tracks)

def is_channel(t, c_val):
    return any([c == c_val for c in t.getChannels()])

def track_sort(t): # sort by 1. variation of pitch, 2. number of notes
    return len(unique_track_notes(t)), len(t.events)

def is_piano_note(pitch):
    return (pitch >= PIANO_RANGE[0]) and (pitch < PIANO_RANGE[1])

def unique_track_notes(t):
    return { e.pitch for e in t.events if e.pitch is not None }

def compress_midi_file(fp, cutoff=6, min_variation=3, supported_types=set([Track.PIANO, Track.PLUCK, Track.BRIGHT])):
    music_file = file2mf(fp)
    
    info_tracks = [t for t in music_file.tracks if not t.hasNotes()]
    note_tracks = [t for t in music_file.tracks if t.hasNotes()]
    
    if len(note_tracks) > cutoff:
        note_tracks = sorted(note_tracks, key=track_sort, reverse=True)
        
    supported_tracks = []
    for idx,t in enumerate(note_tracks):
        if len(supported_tracks) >= cutoff: break
        track_type = get_track_type(t)
        if track_type not in supported_types: continue
        pitch_set = unique_track_notes(t)
        if (len(pitch_set) < min_variation): continue # must have more than x unique notes
        if not all(map(is_piano_note, pitch_set)): continue # must not contain midi notes outside of piano range
#         if track_type == Track.UNDEF: print('Could not designate track:', fp, t)
        change_track_instrument(t, type2inst[track_type])
        supported_tracks.append(t)
    if not supported_tracks: return None
    music_file.tracks = info_tracks + supported_tracks
    return music_file

def get_track_type(t):
    if is_channel(t, 10): return Track.PERC
    i = get_track_instrument(t)
    if i in PIANO_TYPES: return Track.PIANO
    if i in PLUCK_TYPES: return Track.PLUCK
    if i in BRIGHT_TYPES: return Track.BRIGHT
    return Track.UNDEF

def get_track_instrument(t):
    for idx,e in enumerate(t.events):
        if e.type == 'PROGRAM_CHANGE': return e.data
    return None

def change_track_instrument(t, value):
    for idx,e in enumerate(t.events):
        if e.type == 'PROGRAM_CHANGE': e.data = value

def print_music21_instruments():
    for i in range(200):
        try: print(i, music21.instrument.instrumentFromMidiProgram(i))
        except: pass

### Vocab variables

#@title
#specifying data paths 
path = 'debussy'

BPB = 4 # beats per bar
TIMESIG = f'{BPB}/4' # default time signature
PIANO_RANGE = (21, 108)
NOTE_RANGE = (1,127)
VALTSEP = -1 # separator value for numpy encoding
VALTCONT = -2 # numpy value for TCONT - needed for compressing chord array

SAMPLE_FREQ = 4
NOTE_SIZE = 128
DUR_SIZE = (10*BPB*SAMPLE_FREQ)+1 # Max length - 8 bars. Or 16 beats/quarternotes
MAX_NOTE_DUR = (8*BPB*SAMPLE_FREQ)

#tokenizing
BOS = 'xxbos'
PAD = 'xxpad'
EOS = 'xxeos'
MASK = 'xxmask' # Used for BERT masked language modeling. 
#CSEQ = 'xxcseq' # Used for Seq2Seq translation - denotes start of chord sequence
#MSEQ = 'xxmseq' # Used for Seq2Seq translation - denotes start of melody sequence
#S2SCLS = 'xxs2scls' # deprecated
#NSCLS = 'xxnscls' # deprecated
SEP = 'xxsep'
IN = 'xxni'     #null instrument

ACCEP_INS = dict()
ACCEP_INS['Piano'] = 0 
ACCEP_INS['Guitar'] = 1
ACCEP_INS['Bass'] = 2 
ACCEP_INS['WoodwindInstrument'] = 3 
ACCEP_INS['BrassInstrument'] = 4 
ACCEP_INS['StringInstrument'] = 5 
ACCEP_INS['Misc'] = 6 

ACCEP_INS_REV = {v:k for k,v in zip(ACCEP_INS.keys(), ACCEP_INS.values())}

NOTE_TOKS = [f'n{i}' for i in range(NOTE_SIZE)] 
DUR_TOKS = [f'd{i}' for i in range(DUR_SIZE)]
#DONE
INS_TOKS = [f'i{i}' for i in range(len(ACCEP_INS.keys()))]

NOTE_START, NOTE_END = NOTE_TOKS[0], NOTE_TOKS[-1]
DUR_START, DUR_END = DUR_TOKS[0], DUR_TOKS[-1]
INS_START, INS_END = INS_TOKS[0], INS_TOKS[-1]

MTEMPO_SIZE = 10
MTEMPO_OFF = 'mt0'
MTEMPO_TOKS = [f'mt{i}' for i in range(MTEMPO_SIZE)]

SEQType = Enum('SEQType', 'Mask, Sentence, Melody, Chords, Empty')

# Important: SEP token must be last
#DONE
# Important: IN token must be second last

#SPECIAL_TOKS = [BOS, PAD, EOS, S2SCLS, MASK, CSEQ, MSEQ, NSCLS, SEP]
SPECIAL_TOKS = [BOS, PAD, EOS, MASK, IN, SEP] # Important: SEP token must be last



#@title
from fastai.torch_core import ParameterModule
def file2stream(fp):
    if isinstance(fp, music21.midi.MidiFile): return music21.midi.translate.midiFileToStream(fp)
    return music21.converter.parse(fp)

def npenc2stream(arr,bpm=120, instr_list = None):
    "Converts numpy encoding to music21 stream"
    chordarr = npenc2chordarr(np.array(arr)) # 1.
    return chordarr2stream(chordarr,bpm=bpm, instr_list = instr_list) # 2.

# 2.
def stream2chordarr(s, note_size=NOTE_SIZE, sample_freq=SAMPLE_FREQ, max_note_dur=MAX_NOTE_DUR):
    "Converts music21.Stream to 1-  numpy array"
    # assuming 4/4 time
    # note x instrument x pitch
    # FYI: midi middle C value=60
    
    # (AS) TODO: need to order by instruments most played and filter out percussion or include the channel
    highest_time = max(s.flat.getElementsByClass('Note').highestTime, s.flat.getElementsByClass('Chord').highestTime)
    maxTimeStep = round(highest_time * sample_freq)+1
    score_arr = np.zeros((maxTimeStep, len(s.parts), NOTE_SIZE))

    def note_data(pitch, note):
        return (pitch.midi, int(round(note.offset*sample_freq)), int(round(note.duration.quarterLength*sample_freq)))
    
    ins=dict()

    # print('---------------------------------------------------------------')

    for idx,part in enumerate(s.parts):
        
        notes=[]
        iterate = False
        
        for elem in part.flat:

            # Verbose 
            # if isinstance(elem,music21.instrument.Instrument):
            #   print(elem)
            # elif (not isinstance(elem, music21.note.Note)) and (not isinstance(elem, music21.chord.Chord)) and (not isinstance(elem, music21.note.Rest)):
            #   print(elem)

            if isinstance(elem,music21.instrument.Instrument) and (elem.instrumentName is not None):
                # Get the classes for each instrument 
                #Flawed logic
                if elem.instrumentName.replace(" ", "") not in ACCEP_INS.keys():
                  classes = set(elem.classes) - {'Instrument', 'Music21Object', 'object', f'{elem.instrumentName.replace(" ", "")}'}
                else:
                  classes = set(elem.classes) - {'Instrument', 'Music21Object', 'object'}

                # Check for piano 
                if("KeyboardInstrument" in classes):
                  ins[idx] = 'Piano'
                  iterate = True 
                # Handle for guitar and bass
                elif(elem.instrumentName == 'Guitar' or elem.instrumentName == 'Acoustic Guitar' or elem.instrumentName == 'Electric Guitar'):
                  ins[idx] = 'Guitar'
                  iterate = True
                elif('Guitar' in classes and ('Bass' in elem.instrumentName)):
                  ins[idx] = 'Bass'
                  iterate = True
                # Handle for remaining instruments 
                elif(len(classes.intersection(ACCEP_INS.keys())) != 0):
                  inter = list(classes.intersection(ACCEP_INS.keys()))
                  try:
                    assert len(inter) <= 1
                  except AssertionError:
                    print('Intersection with ACCEP_INS have multiple values: ',inter)
                  ins[idx] = inter[0]
                  iterate = True
                else:
                  # print(f'instrument rejected : {elem.instrumentName}')
                  break

                # if elem.instrumentName in ACCEP_INS.keys():
                #     ins[idx] = elem.instrumentName 
                #     iterate = True
                # else :
                #     print(f'instrument rejected : {elem.instrumentName}')
                #     break
            elif isinstance(elem,music21.instrument.Instrument) and (elem.instrumentName is  None):
                ins[idx] = 'Misc'
                iterate = True 
            
            if isinstance(elem, music21.note.Note):
                notes.append(note_data(elem.pitch, elem))
            if isinstance(elem, music21.chord.Chord):
                for p in elem.pitches:
                    notes.append(note_data(p, elem)) 
        
        # print('---------------------------------------------------------------')

        # sort notes by offset (1), duration (2) so that hits are not overwritten and longer notes have priority
        notes_sorted = sorted(notes, key=lambda x: (x[1], x[2])) 
        
        if(iterate == True):
            for n in notes_sorted:
                if n is None: continue
                pitch,offset,duration = n
                if max_note_dur is not None and duration > max_note_dur: duration = max_note_dur
                score_arr[offset,idx, pitch] = duration
                score_arr[offset+1:offset+duration, idx, pitch] = VALTCONT      # Continue holding not
    
    return score_arr, ins

def chordarr2npenc(chordarr, skip_last_rest=True):
    # combine instruments
    # print(chordarr)

    result = []
    wait_count = 0
    for idx,timestep in enumerate(chordarr):
        flat_time = timestep2npenc(timestep)
        #DONE
        #print(idx, flat_time)
        if len(flat_time) == 0:
            wait_count += 1
        else:
            # pitch, octave, duration, instrument
            # DONE: Replaced -2 with (-2 - len(NOTE_TOKS) - len(DUR_TOKS)) so that in `npenc2idxenc`,
            # `t[:, 2] = t[:, 2] + vocab.ins_range[0]` gives right mapping in vocal.itos
            if wait_count > 0: result.append([VALTSEP, wait_count, -2 - len(NOTE_TOKS) - len(DUR_TOKS)])
            result.extend(flat_time)
            wait_count = 1
    if wait_count > 0 and not skip_last_rest: result.append([VALTSEP, wait_count, -2 - len(NOTE_TOKS) - len(DUR_TOKS)])
    return np.array(result,dtype = int)
    #return np.array(result, dtype=int).reshape(-1, 2) # reshaping. Just in case result is empty

'''
def chordarr2npenc(chordarr, skip_last_rest=True):
    # combine instruments
    result = []
    wait_count = 0
    for idx,timestep in enumerate(chordarr):
        flat_time = timestep2npenc(timestep)
        if len(flat_time) == 0:
            wait_count += 1
        else:
            # pitch, octave, duration, instrument
            if wait_count > 0: result.append([VALTSEP, wait_count])
            result.extend(flat_time)
            wait_count = 1
    if wait_count > 0 and not skip_last_rest: result.append([VALTSEP, wait_count])
    return np.array(result, dtype=int).reshape(-1, 2) # reshaping. Just in case result is empty
'''

# Note: not worrying about overlaps - as notes will still play. just look tied
# http://web.mit.edu/music21/doc/moduleReference/moduleStream.html#music21.stream.Stream.getOverlaps
def timestep2npenc(timestep, note_range=NOTE_RANGE, enc_type='full'):
    
    # inst x pitch
    notes = []
    for i,n in zip(*timestep.nonzero()):
        d = timestep[i,n]
        if d < 0: continue # only supporting short duration encoding for now
        if n < note_range[0] or n >= note_range[1]: continue # must be within midi range
        notes.append([n,d,i])
        
    notes = sorted(notes, key=lambda x: x[0], reverse=True) # sort by note (highest to lowest)
    
    if enc_type is None: 
        # note, duration
        return [n[:2] for n in notes] 
    if enc_type == 'parts':
        # note, duration, part
        return [n for n in notes]
    if enc_type == 'full':
        # note_class, duration , instrument
        return [[n, d, i] for n,d,i in notes] 

'''
# Note: not worrying about overlaps - as notes will still play. just look tied
# http://web.mit.edu/music21/doc/moduleReference/moduleStream.html#music21.stream.Stream.getOverlaps
def timestep2npenc(timestep, note_range=PIANO_RANGE, enc_type=None):
    # inst x pitch
    notes = []
    a, b = zip(*timestep.nonzero())
    for i,n in zip(*timestep.nonzero()):
        d = timestep[i,n]
        if d < 0: continue # only supporting short duration encoding for now
        if n < note_range[0] or n >= note_range[1]: continue # must be within midi range
        notes.append([n,d,i])
        
    notes = sorted(notes, key=lambda x: x[0], reverse=True) # sort by note (highest to lowest)
    
    if enc_type is None: 
        # note, duration
        return [n[:2] for n in notes] 
    if enc_type == 'parts':
        # note, duration, part
        return [n for n in notes]
    if enc_type == 'full':
        # note_class, duration, octave, instrument
        return [[n%12, d, n//12, i] for n,d,i in notes] 
'''

### **Decoding Functions**

ACCEP_INS.keys()

#@title
# 1.
def npenc2chordarr(npenc,note_size=NOTE_SIZE):
    # print(npenc.shape)
    num_instruments = 1 if npenc.shape[1] <= 2 else npenc.max(axis=0)[-1]
    # print(num_instruments)
    # num_instruments = 7

    max_len = npenc_len(npenc)
    # score_arr = (steps, inst, note)
    try:
      score_arr = np.zeros((max_len, num_instruments + 1, note_size))
    except:
      print(npenc)
      print(max_len, num_instruments, note_size)
      raise ValueError

    idx = 0
    for step in npenc:
        n,d,i = (step.tolist()+[0])[:3] # or n,d,i
        if n < VALTSEP: continue # special token
        if n == VALTSEP:
            idx += d
            continue
        score_arr[idx,i,n] = d
    return score_arr

def npenc_len(npenc):
    duration = 0
    for t in npenc:
        if t[0] == VALTSEP: duration += t[1]
    return duration + 1


# 2.
def chordarr2stream(arr,sample_freq=SAMPLE_FREQ, bpm=120, instr_list = None):
    duration = music21.duration.Duration(1. / sample_freq)
    stream = music21.stream.Score()
    stream.append(music21.meter.TimeSignature(TIMESIG))
    stream.append(music21.tempo.MetronomeMark(number=bpm))
    stream.append(music21.key.KeySignature(0))
    for inst in range(arr.shape[1]):
        p = partarr2stream(arr[:,inst,:],inst,duration)
        #print(p.getInstrument())
        if instr_list is not None and str(p.getInstrument()) not in instr_list:
          #print('+', instr_list)
          continue
        stream.append(p)
    stream = stream.transpose(0)
    return stream

# 2b.
def partarr2stream(partarr,inst,duration):
    "convert instrument part to music21 chords"
#    part = music21.stream.Part()
#    part.append(music21.instrument.Piano())
#    part_append_duration_notes(partarr, duration, part) # notes already have duration calculated
    l = len(ACCEP_INS_REV) 
    inst = inst%l
    part = music21.stream.Part()
    if(ACCEP_INS_REV[inst] == 'Piano'):
        part.append(music21.instrument.Piano())
    #DONE
    elif(ACCEP_INS_REV[inst] == 'Bass'):
        part.append(music21.instrument.AcousticBass())
    elif(ACCEP_INS_REV[inst] == 'Guitar'):
        part.append(music21.instrument.AcousticGuitar())  
    elif(ACCEP_INS_REV[inst] == 'WoodwindInstrument'):
        part.append(music21.instrument.TenorSaxophone())
    elif(ACCEP_INS_REV[inst] == 'BrassInstrument'):
        part.append(music21.instrument.Trumpet())   
    
    elif(ACCEP_INS_REV[inst] == 'Trumpet'):
        part.append(music21.instrument.Trumpet())
    elif(ACCEP_INS_REV[inst] == 'Tenor Saxophone'):
        part.append(music21.instrument.TenorSaxophone())
    elif(ACCEP_INS_REV[inst] == 'Vibraphone'):
        part.append(music21.instrument.Vibraphone())
    elif(ACCEP_INS_REV[inst] == 'Baritone Saxophone'):
        part.append(music21.instrument.BaritoneSaxophone())
    elif(ACCEP_INS_REV[inst] == 'Acoustic Bass'):
        part.append(music21.instrument.AcousticBass())
    elif(ACCEP_INS_REV[inst] == 'Trombone'):
        part.append(music21.instrument.Trombone())
    elif(ACCEP_INS_REV[inst] == 'Flute'):
        part.append(music21.instrument.Flute())
    elif(ACCEP_INS_REV[inst] == 'Saxophone'):
        part.append(music21.instrument.Saxophone())
    elif(ACCEP_INS_REV[inst] == 'Electric Bass'):
        part.append(music21.instrument.ElectricBass())
    elif(ACCEP_INS_REV[inst] == 'Electric Guitar'):
        part.append(music21.instrument.ElectricGuitar())
    elif(ACCEP_INS_REV[inst] == 'Acoustic Guitar'):
        part.append(music21.instrument.AcousticGuitar())
    elif(ACCEP_INS_REV[inst] == 'Glockenspiel'):
        part.append(music21.instrument.Glockenspiel())
    elif(ACCEP_INS_REV[inst] == 'Vibraphone'):
        part.append(music21.instrument.Vibraphone())
    elif(ACCEP_INS_REV[inst] == 'Violin'):
        part.append(music21.instrument.Violin())
    else:
        part.append(music21.instrument.Piano())
    part_append_duration_notes(partarr, duration, part)
    

    return part

def part_append_duration_notes(partarr, duration, stream):
    "convert instrument part to music21 chords"
    for tidx,t in enumerate(partarr):
        note_idxs = np.where(t > 0)[0] # filter out any negative values (continuous mode)
        if len(note_idxs) == 0: continue
        notes = []
        for nidx in note_idxs:
            note = music21.note.Note(nidx)
            note.duration = music21.duration.Duration(partarr[tidx,nidx]*duration.quarterLength)
            notes.append(note)
        for g in group_notes_by_duration(notes):
            if len(g) == 1:
                stream.insert(tidx*duration.quarterLength, g[0])
            else:
                chord = music21.chord.Chord(g)
                stream.insert(tidx*duration.quarterLength, chord)
    return stream

from itertools import groupby
#  combining notes with different durations into a single chord may overwrite conflicting durations. Example: aylictal/still-waters-run-deep
def group_notes_by_duration(notes):
    "separate notes into chord groups"
    keyfunc = lambda n: n.duration.quarterLength
    notes = sorted(notes, key=keyfunc)
    return [list(g) for k,g in groupby(notes, keyfunc)]


# Midi -> npenc Conversion helpers
def is_valid_npenc(npenc, note_range=PIANO_RANGE, max_dur=DUR_SIZE, 
                   min_notes=32, input_path=None, verbose=True):
    if len(npenc) < min_notes:
        if verbose: print('Sequence too short:', len(npenc), input_path)
        return False
    if (npenc[:,1] >= max_dur).any(): 
        if verbose: print(f'npenc exceeds max {max_dur} duration:', npenc[:,1].max(), input_path)
        return False
    # https://en.wikipedia.org/wiki/Scientific_pitch_notation - 88 key range - 21 = A0, 108 = C8
    if ((npenc[...,0] > VALTSEP) & ((npenc[...,0] < note_range[0]) | (npenc[...,0] >= note_range[1]))).any(): 
        print(f'npenc out of piano note range {note_range}:', input_path)
        return False
    return True

# seperates overlapping notes to different tracks
def remove_overlaps(stream, separate_chords=True):
    if not separate_chords:
        return stream.flat.makeVoices().voicesToParts()
    return separate_melody_chord(stream)

# seperates notes and chords to different tracks
def separate_melody_chord(stream):
    new_stream = music21.stream.Score()
    if stream.timeSignature: new_stream.append(stream.timeSignature)
    new_stream.append(stream.metronomeMarkBoundaries()[0][-1])
    if stream.keySignature: new_stream.append(stream.keySignature)
    
    melody_part = music21.stream.Part(stream.flat.getElementsByClass('Note'))
    melody_part.insert(0, stream.getInstrument())
    chord_part = music21.stream.Part(stream.flat.getElementsByClass('Chord'))
    chord_part.insert(0, stream.getInstrument())
    new_stream.append(melody_part)
    new_stream.append(chord_part)
    return new_stream
    
 # processing functions for sanitizing data

def compress_chordarr(chordarr):
    return shorten_chordarr_rests(trim_chordarr_rests(chordarr))

def trim_chordarr_rests(arr, max_rests=4, sample_freq=SAMPLE_FREQ):
    # max rests is in quarter notes
    # max 1 bar between song start and end
    start_idx = 0
    max_sample = max_rests*sample_freq
    for idx,t in enumerate(arr):
        if (t != 0).any(): break
        start_idx = idx+1
        
    end_idx = 0
    for idx,t in enumerate(reversed(arr)):
        if (t != 0).any(): break
        end_idx = idx+1
    start_idx = start_idx - start_idx % max_sample
    end_idx = end_idx - end_idx % max_sample
#     if start_idx > 0 or end_idx > 0: print('Trimming rests. Start, end:', start_idx, len(arr)-end_idx, end_idx)
    return arr[start_idx:(len(arr)-end_idx)]

def shorten_chordarr_rests(arr, max_rests=8, sample_freq=SAMPLE_FREQ):
    # max rests is in quarter notes
    # max 2 bar pause
    rest_count = 0
    result = []
    max_sample = max_rests*sample_freq
    for timestep in arr:
        if (timestep==0).all(): 
            rest_count += 1
        else:
            if rest_count > max_sample:
#                 old_count = rest_count
                rest_count = (rest_count % sample_freq) + max_sample
#                 print(f'Compressing rests: {old_count} -> {rest_count}')
            for i in range(rest_count): result.append(np.zeros(timestep.shape))
            rest_count = 0
            result.append(timestep)
    for i in range(rest_count): result.append(np.zeros(timestep.shape))
    return np.array(result)

# sequence 2 sequence convenience functions

def stream2npenc_parts(stream, sort_pitch=True):
    chordarr = stream2chordarr(stream)
    _,num_parts,_ = chordarr.shape
    parts = [part_enc(chordarr, i) for i in range(num_parts)]
    return sorted(parts, key=avg_pitch, reverse=True) if sort_pitch else parts

def chordarr_combine_parts(parts):
    max_ts = max([p.shape[0] for p in parts])
    parts_padded = [pad_part_to(p, max_ts) for p in parts]
    print(parts_padded[0].shape, parts_padded[1].shape)
    chordarr_comb = np.concatenate(parts_padded, axis=0)
    return chordarr_comb

def pad_part_to(p, target_size):
    pad_width = ((0,target_size-p.shape[0]),(0,0),(0,0))
    return np.pad(p, pad_width, 'constant')

def part_enc(chordarr, part):
    partarr = chordarr[:,part:part+1,:]
    npenc = chordarr2npenc(partarr)
    return npenc

def avg_tempo(t, sep_idx=VALTSEP):
    avg = t[t[:, 0] == sep_idx][:, 1].sum()/t.shape[0]
    avg = int(round(avg/SAMPLE_FREQ))
    return 'mt'+str(min(avg, MTEMPO_SIZE-1))

def avg_pitch(t, sep_idx=VALTSEP):
    return t[t[:, 0] > sep_idx][:, 0].mean()   

###Extra fastai utilities

#@title
def check_valid_ins(ins):
  count = 0
  ls = list(set(val for val in ins.values()))
  for i in ls:
    if i == 'Piano':
      count+= 1
    elif i == 'Acoustic Bass' or i == 'Electric Bass':
      count += 1
    elif i == 'Acoustic Guitar' or i == 'Electric Guitar':
      count += 1
    elif i == 'Violin':
      count += 1
    elif i == 'Saxophone':
      count += 1
  if(count>=3):
    return True
  return False

import time 

#DONE
def fastai_num_track_filter (arg, num_ins_thresh = 1):
  global time_taken_avg, processed_files

  ins1 = 'Piano'
  ins2 = 'Bass'

  t1 = time.time()
  # print('-> Processing file no. ', files_dict[arg])
  # Try for inconsistent vocab and file errors  
  try:
    filename, file_extension = os.path.splitext(arg)
    if file_extension == '.mid':
      item = MusicItem.from_file(arg, data_vocab)
      data_vocab.textify(item.data)
    elif file_extension == '.npy':
      nparr = np.load(arg, allow_pickle=True)
      item = MusicItem.from_npenc(nparr, data_vocab)

      data_vocab.textify(item.data)
  except:
    print('\t file discarded : ', arg)
    os.makedirs('/content/drive/MyDrive/datasets/discarded', exist_ok = True) 
    shutil.move(arg, os.path.join('/content/drive/MyDrive/datasets/discarded',os.path.basename(arg)))

    # t2 = time.time()
    # time_taken_avg = ((t2 - t1) + time_taken_avg*processed_files)/(processed_files + 1)
    # processed_files += 1

    # print(f'\t estimated_time : {time_taken_avg*len(files_dict)}')
    return False 
  
  # t2 = time.time()
  # time_taken_avg = ((t2 - t1) + time_taken_avg*processed_files)/(processed_files + 1)
  # processed_files += 1
  
  # Check for no. of instruments 
  # print(item.ins)

  try:
    if file_extension == '.mid':
      mt_item = MultitrackItem.from_file(arg, data_vocab, first_ins = ins1, second_ins = ins2)
      data_vocab.textify(mt_item.first_instrument)
      data_vocab.textify(mt_item.second_instrument)
    elif file_extension == '.npy':
      nparr = np.load(arg, allow_pickle=True)
      mt_item = MultitrackItem.from_npenc(nparr, data_vocab, first_ins = ins1, second_ins = ins2)
  except:
    print(f'\t file discarded due to not containing overlapping notes from a {ins1} or {ins2} track : {arg}')
    return False


  if (item.ins is not None) and (len(item.ins.keys()) >= num_ins_thresh):
    print('\t file accepted : ', arg)
    print(f'\t {item.ins}')
    # print(f'\t estimated_time : {time_taken_avg*len(files_dict)}')
    return True
  #Else if we did not store the track -> instrument dictionary from MIDI file
  elif item.ins is None:
    # print(item.data.shape)
    lst =  list(item.data)
    # print(lst)
    cond_lst = [True if ( (x >= data_vocab.ins_range[0] and x < data_vocab.ins_range[1]) or (x == data_vocab.stoi['xxni']) ) else False for x in lst]
    ins_idxs = item.data[cond_lst]
    # print([data_vocab.itos[x] for index,x in enumerate(lst) if index%3 == 0])
    uniq_ins = np.unique(ins_idxs)
    num_ins = len(uniq_ins)
    if num_ins >= num_ins_thresh:

      print(f'\t {[data_vocab.itos[x] for x in uniq_ins]}')
      
      # Check if song contains piano and bass tracks 
      count = 0 
      for x in uniq_ins:
        ins = data_vocab.itos[x] 

        ins1_idx = 'i' + str(ACCEP_INS[ins1])
        ins2_idx = 'i' + str(ACCEP_INS[ins2])
        if(ins in [ins1_idx, ins2_idx]): count += 1 

      if(count == 2): print('\t file accepted : ', arg) ; return True
      else : print(f'\t file discarded due to not containing either a {ins1} or a {ins2} track : {arg}') ; return False
    else:
      print('\t file discarded due to less instruments : ', arg)
      return False

  else:
    print('\t file discarded due to less instruments : ', arg)
    # print(f'\t estimated_time : {time_taken_avg*len(files_dict)}')
    return False

  # ins = dict()

  # if not is_empty_midi(arg):
  #   s = file2stream(arg)
    
  #   for idx,part in enumerate(s.parts):
  #     for elem in part.flat:
  #       if (isinstance(elem,music21.instrument.Instrument)) and (elem.instrumentName is not None): 
  #         # DONE
  #         # Get the classes for each instrument 
  #         if elem.instrumentName.replace(" ", "") not in ACCEP_INS.keys():
  #           classes = set(elem.classes) - {'Instrument', 'Music21Object', 'object', f'{elem.instrumentName.replace(" ", "")}'}
  #         else:
  #           classes = set(elem.classes) - {'Instrument', 'Music21Object', 'object'}
          
  #         # Check for piano 
  #         if("KeyboardInstrument" in classes):
  #           if('Piano' not in ins.keys()): ins['Piano'] = 1
  #           else: ins['Piano'] += 1
  #         # Handle for guitar and bass
  #         elif(elem.instrumentName == 'Guitar' or elem.instrumentName == 'Acoustic Guitar' or elem.instrumentName == 'Electric Guitar'):
  #           if('Guitar' not in ins.keys()): ins['Guitar'] = 1
  #           else: ins['Guitar'] += 1
  #         elif('Guitar' in classes and ('Bass' in elem.instrumentName)):
  #           if('Bass' not in ins.keys()): ins['Bass'] = 1
  #           else: ins['Bass'] += 1
  #         # Handle for remaining instruments 
  #         else:
  #           inter = list(classes.intersection(ACCEP_INS.keys()))
  #           if(len(inter) != 0):
  #             if(inter[0] not in ins.keys()): ins[inter[0]] = 1
  #             else: ins[inter[0]] += 1
  #       elif isinstance(elem,music21.instrument.Instrument) and (elem.instrumentName is None):
  #           if('Misc' not in ins.keys()): ins['Misc'] = 1
  #           else: ins['Misc'] += 1
  #       else: 
  #         break 
  #         # if elem.instrumentName in (ACCEP_INS.keys()):
  #         #     ins_count += 1
  #         # else :
  #         #     break


## **Transformer-XL**

####TODO

#@title
'''
(i) Correct the filter function : fast_ai_filter_function 
(ii) filter the dataset / enforce max duration + notes + instruments in : npenc2idxenc 
(ii.v) Add xxbos, xxeos, xxpad etc

-> Balance the data (currently imbalanced towards piano, which also might be the cause for 'dense' outputs for piano in terms of num. notes each timestep)
-> One thing that could help with the above is to group multiple instruments together, eg: merging AcousticGuitar, ElectricGuitar, BassGuitar into StringInstrument
-> Filter output of model properly (.to_stream() fails many times due to corner cases violating 'partnerless' n or d or i
-> Fix the flawed logic of conversion of every song to 4/4 time signature by rounding off, and then add time signature tokens to model vocab 
  and dataset in order to get consistent output.


(iv) train :)
'''

###**vocab**

#@title
#SEE 'Vocab variables'for more details

# Vocab - token to index mapping
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
        itos = SPECIAL_TOKS + NOTE_TOKS + DUR_TOKS + INS_TOKS #+ MTEMPO_TOKS
        
        # if len(itos)%8 != 0:
        #     itos = itos + [f'dummy{i}' for i in range(len(itos)%8)]
        return cls(itos)
    
    @classmethod
    def load(cls, path):
        "Load the `Vocab` contained in `path`"
        itos = pickle.load(open(path, 'rb'))
        return cls(itos)

###**dataloader.py**

#@title
import fastai

#@title
#https://github.com/bearpelican/musicautobot/blob/master/musicautobot/music_transformer/dataloader.py

"Fastai Language Model Databunch modified to work with music"
from fastai.basics import *
# from fastai.basic_data import DataBunch
from fastai.text.data import LMLabelList
#from .transform import *
#from ..vocab import MusicVocab


class MusicDataBunch(DataBunch):
    "Create a `TextDataBunch` suitable for training a language model."
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', no_check:bool=False, bs=64, val_bs:int=None, 
               num_workers:int=0, device:torch.device=None, collate_fn:Callable=data_collate, 
               dl_tfms:Optional[Collection[Callable]]=None, bptt:int=70,
               preloader_cls=None, shuffle_dl=False, transpose_range=None, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` in `path` from the `datasets` for language modelling."
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        preloader_cls = MusicPreloader if preloader_cls is None else preloader_cls
        val_bs = ifnone(val_bs, bs)
        datasets = [preloader_cls(ds, shuffle=(i==0), bs=(bs if i==0 else val_bs), bptt=bptt, transpose_range=transpose_range, **kwargs) 
                    for i,ds in enumerate(datasets)]
        val_bs = bs
        dl_tfms = [partially_apply_vocab(tfm, train_ds.vocab) for tfm in listify(dl_tfms)]
        dls = [DataLoader(d, b, shuffle=shuffle_dl) for d,b in zip(datasets, (bs,val_bs,val_bs,val_bs)) if d is not None]
        return cls(*dls, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)
    
    @classmethod    
    def from_folder(cls, path:PathOrStr, extensions='.npy', **kwargs):
        files = get_files(path, extensions=extensions, recurse=True);
        return cls.from_files(files, path, **kwargs)
    
    @classmethod
    def from_files(cls, files, path, processors=None, split_pct=0.1, 
                   vocab=None, list_cls=None, **kwargs):
        if vocab is None: vocab = MusicVocab.create()
        if list_cls is None: list_cls = MusicItemList
        src = (list_cls(items=files, path=path, processor=processors, vocab=vocab)
                .filter_by_func(fastai_num_track_filter)
                .split_by_rand_pct(split_pct, seed=6)
                .label_const(label_cls=LMLabelList))
        return src.databunch(**kwargs)

    @classmethod
    def empty(cls, path, **kwargs):
        vocab = MusicVocab.create()
        src = MusicItemList([], path=path, vocab=vocab, ignore_empty=True).split_none()
        return src.label_const(label_cls=LMLabelList).databunch()
        
def partially_apply_vocab(tfm, vocab):
    if 'vocab' in inspect.getfullargspec(tfm).args:
        return partial(tfm, vocab=vocab)
    return tfm
    
class MusicItemList(ItemList):
    _bunch = MusicDataBunch
    
    def __init__(self, items:Iterator, vocab:MusicVocab=None, **kwargs):
        super().__init__(items, **kwargs)
        self.vocab = vocab
        self.copy_new += ['vocab']
    
    def get(self, i):
        o = super().get(i)
        if is_pos_enc(o): 
            return MusicItem.from_idx(o, self.vocab)
        return MusicItem(o, self.vocab)

def is_pos_enc(idxenc):
    if len(idxenc.shape) == 2 and idxenc.shape[0] == 2: return True
    return idxenc.dtype == np.object and idxenc.shape == (2,)

class MusicItemProcessor(PreProcessor):
    "`PreProcessor` that transforms numpy files to indexes for training"
    def process_one(self,item):
        item = MusicItem.from_npenc(item, vocab=self.vocab)
        return item.to_idx()
    
    def process(self, ds):
        self.vocab = ds.vocab
        super().process(ds)
        
class OpenNPFileProcessor(PreProcessor):
    "`PreProcessor` that opens the filenames and read the texts."
    def process_one(self,item):
        return np.load(item, allow_pickle=True) if isinstance(item, Path) else item

class Midi2ItemProcessor(PreProcessor):
    "Skips midi preprocessing step. And encodes midi files to MusicItems"
    def process_one(self,item):
        # print('Midi2ItemProcess process_one')
        item = MusicItem.from_file(item, vocab=self.vocab)
        print('item.to_idx(): ', item.to_idx())
        return item.to_idx()
    
    def process(self, ds):
        self.vocab = ds.vocab
        super().process(ds)

## For npenc dataset
class MusicPreloader(Callback):
    "Transforms the tokens in `dataset` to a stream of contiguous batches for language modelling."
    
    class CircularIndex():
        "Handles shuffle, direction of indexing, wraps around to head tail in the ragged array as needed"
        def __init__(self, length:int, forward:bool): self.idx, self.forward = np.arange(length), forward
        def __getitem__(self, i): 
            # print('MusicPreloader __getitem__ ')
            
            return self.idx[ i%len(self.idx) if self.forward else len(self.idx)-1-i%len(self.idx)]
        def __len__(self) -> int: return len(self.idx)
        def shuffle(self): np.random.shuffle(self.idx)

    def __init__(self, dataset:LabelList, lengths:Collection[int]=None, bs:int=32, bptt:int=70, backwards:bool=False, 
                 shuffle:bool=False, y_offset:int=1, 
                 transpose_range=None, transpose_p=0.5,
                 encode_position=True,
                 **kwargs):
        self.dataset,self.bs,self.bptt,self.shuffle,self.backwards,self.lengths = dataset,bs,bptt,shuffle,backwards,lengths
        self.vocab = self.dataset.vocab
        self.bs *= num_distrib() or 1
        self.totalToks,self.ite_len,self.idx = int(0),None,None
        self.y_offset = y_offset
        
        self.transpose_range,self.transpose_p = transpose_range,transpose_p
        self.encode_position = encode_position
        self.bptt_len = self.bptt
        
        self.allocate_buffers() # needed for valid_dl on distributed training - otherwise doesn't get initialized on first epoch

    def __len__(self): 
        if self.ite_len is None:
            if self.lengths is None: self.lengths = np.array([len(item) for item in self.dataset.x])
            self.totalToks = self.lengths.sum()
            self.ite_len   = self.bs*int( math.ceil( self.totalToks/(self.bptt*self.bs) )) if self.item is None else 1
        return self.ite_len

    def __getattr__(self,k:str)->Any: return getattr(self.dataset, k)
   
    def allocate_buffers(self):
        "Create the ragged array that will be filled when we ask for items."
        if self.ite_len is None: len(self)
        self.idx   = MusicPreloader.CircularIndex(len(self.dataset.x), not self.backwards)
        
        # batch shape = (bs, bptt, 2 - [index, pos]) if encode_position. Else - (bs, bptt)
        buffer_len = (2,) if self.encode_position else ()
        self.batch = np.zeros((self.bs, self.bptt+self.y_offset) + buffer_len, dtype=np.int64)
        self.batch_x, self.batch_y = self.batch[:,0:self.bptt], self.batch[:,self.y_offset:self.bptt+self.y_offset] 
        #ro: index of the text we're at inside our datasets for the various batches
        self.ro    = np.zeros(self.bs, dtype=np.int64)
        #ri: index of the token we're at inside our current text for the various batches
        self.ri    = np.zeros(self.bs, dtype=np.int)
        
        # allocate random transpose values. Need to allocate this before hand.
        self.transpose_values = self.get_random_transpose_values()
        
    def get_random_transpose_values(self):
        if self.transpose_range is None: return None
        n = len(self.dataset)
        rt_arr = torch.randint(*self.transpose_range, (n,))-self.transpose_range[1]//2
        mask = torch.rand(rt_arr.shape) > self.transpose_p
        rt_arr[mask] = 0
        return rt_arr

    def on_epoch_begin(self, **kwargs):
        if self.idx is None: self.allocate_buffers()
        elif self.shuffle:   
            self.ite_len = None
            self.idx.shuffle()
            self.transpose_values = self.get_random_transpose_values()
            self.bptt_len = self.bptt
        self.idx.forward = not self.backwards 

        step = self.totalToks / self.bs
        ln_rag, countTokens, i_rag = 0, 0, -1
        for i in range(0,self.bs):
            #Compute the initial values for ro and ri 
            while ln_rag + countTokens <= int(step * i):
                countTokens += ln_rag
                i_rag       += 1
                ln_rag       = self.lengths[self.idx[i_rag]]
            self.ro[i] = i_rag
            self.ri[i] = ( ln_rag - int(step * i - countTokens) ) if self.backwards else int(step * i - countTokens)
        
    #Training dl gets on_epoch_begin called, val_dl, on_epoch_end
    def on_epoch_end(self, **kwargs): self.on_epoch_begin()

    def __getitem__(self, k:int):
        j = k % self.bs
        if j==0:
            if self.item is not None: return self.dataset[0]
            if self.idx is None: self.on_epoch_begin()
                
        self.ro[j],self.ri[j] = self.fill_row(not self.backwards, self.dataset.x, self.idx, self.batch[j][:self.bptt_len+self.y_offset], 
                                              self.ro[j], self.ri[j], overlap=1, lengths=self.lengths)
        return self.batch_x[j][:self.bptt_len], self.batch_y[j][:self.bptt_len]

    def fill_row(self, forward, items, idx, row, ro, ri, overlap, lengths):
        "Fill the row with tokens from the ragged array. --OBS-- overlap != 1 has not been implemented"
        ibuf = n = 0 
        ro  -= 1
        while ibuf < row.shape[0]:  
            ro   += 1 
            ix    = idx[ro]
            
            item = items[ix]
            if self.transpose_values is not None: 
                item = item.transpose(self.transpose_values[ix].item())
                
            if self.encode_position:
                # Positions are colomn stacked with indexes. This makes it easier to keep in sync
                rag = np.stack([item.data, item.position], axis=1)
            else:
                rag = item.data
                
            if forward:
                ri = 0 if ibuf else ri
                n  = min(lengths[ix] - ri, row.shape[0] - ibuf)
                row[ibuf:ibuf+n] = rag[ri:ri+n]
            else:    
                ri = lengths[ix] if ibuf else ri
                n  = min(ri, row.size - ibuf) 
                row[ibuf:ibuf+n] = rag[ri-n:ri][::-1]
            ibuf += n
        return ro, ri + ((n-overlap) if forward else -(n-overlap))



def batch_position_tfm(b):
    "Batch transform for training with positional encoding"
    x,y = b
    x = {
        'x': x[...,0],
        'pos': x[...,1]
    }
    return x, y[...,0]

###**Music Item**

#@title
#https://github.com/bearpelican/musicautobot/blob/master/musicautobot/music_transformer/transform.py

#from ..numpy_encode import *
import numpy as np
from enum import Enum
import torch
#from ..vocab import *
from functools import partial

SEQType = Enum('SEQType', 'Mask, Sentence, Melody, Chords, Empty')

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
    def from_npenc(cls, npenc, vocab, stream=None, ins = None): return MusicItem(npenc2idxenc(npenc, vocab, ins = ins), vocab, ins = ins, stream = stream)
    
    @classmethod
    def from_idx(cls, item, vocab):
        idx,pos = item
        return MusicItem(idx, vocab=vocab, position=pos)
    def to_idx(self): return self.data, self.position
    
    @classmethod
    def empty(cls, vocab, seq_type=SEQType.Sentence):
        return MusicItem(seq_prefix(seq_type, vocab), vocab)

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

    def to_individual_instrument(self, ins='Piano'):
        return MusicItem(filter_by_ins(self.data, self.vocab, ACCEP_INS[ins]), self.vocab, ins = ins, stream = self.stream)
      
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

    #DONE
    def trim_bw_beat(self, beat_low, beat_high, include_last_sep=False):
        return self.new(trim_bw_beat(self.data, self.position, self.vocab, beat_low, beat_high, include_last_sep))
    
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

# Filter by instruments 
def filter_by_ins(t, vocab, ins):
    npenc = idxenc2npenc(t, vocab)
    
    # print(np.unique(npenc[:,2]))
    # print(npenc[:20])
    
    filter_ins_list = list(ACCEP_INS.values())
    filter_ins_list.remove(ins)

    # print(filter_ins_list)

    for i in filter_ins_list:
      npenc = npenc[np.where(npenc[:,2] != i)]

    # print(np.unique(npenc[:,2]))
    # print(npenc[:20])

    return npenc2idxenc(npenc, vocab)

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
def npenc2idxenc(t, vocab, ins = None, seq_type=SEQType.Sentence, add_eos=True):
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
        

        prefix = seq_prefix(seq_type, vocab)
        suffix = np.array([vocab.stoi[EOS]]) if add_eos else np.empty(0, dtype=int)
        # print('npenc2idxenc result: ',np.concatenate([t.reshape(-1), suffix]))
    except IndexError as e:
        print('IndexError, t.shape:', t.shape )
        raise e
    return np.concatenate([prefix, t.reshape(-1), suffix])

def seq_prefix(seq_type, vocab):
    if seq_type == SEQType.Empty: return np.empty(0, dtype=int)
    start_token = vocab.bos_idx
    if seq_type == SEQType.Chords: start_token = vocab.stoi[CSEQ]
    if seq_type == SEQType.Melody: start_token = vocab.stoi[MSEQ]
    return np.array([start_token, vocab.pad_idx])

#IMP
#TODO: Use MusicItem/self.ins to verify proper reconversion into npenc of shape [X, 3]
def idxenc2npenc(t, vocab, validate=True):
    # if validate: 
    #   t = to_valid_idxenc(t, vocab.npenc_range)
      
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
    try:
      last_idx_tok_idx_rev = ins_toks[::-1].index(True)
    except ValueError:
      print('ValueError: ins_toks = ', ins_toks)
    # print(last_idx_tok_idx_rev)
    # print('Last elements before proc: ', t[-5:])
    print('t before clipping to last ins token: ', [vocab.itos[elem] for elem in t])
    
    note_toks = [True if vocab.is_note(x) else False for x in t]
    print('note_toks: ', note_toks)
    try:
      first_note_tok= note_toks.index(True)
    except ValueError:
      print('ValueError: note_toks = ', note_toks)

    print('first_note_tok: ', first_note_tok)

    t = t[first_note_tok :(len(ins_toks) - last_idx_tok_idx_rev)]
    # print('Last elements after proc: ', t[-5:])

    print('ins toks: ',ins_toks)
    print('last index token: ', last_idx_tok_idx_rev )
    print('t after clipping to last ins token: ', [vocab.itos[elem] for elem in t])
    print('length of t after clipping to last ins token: ',len(t))


    try:
      t = t.copy().reshape(-1, 3)
    except:
      raise IndexError

    # print('idxenc2npenc input: ', t[:,0], t[:,1], t[:,2]) 

    

    if t.shape[0] == 0: return t
        
    t[:, 0] = t[:, 0] - vocab.note_range[0]
    t[:, 1] = t[:, 1] - vocab.dur_range[0]
    t[:, 2] = t[:, 2] - vocab.ins_range[0]

    # print('idxenc2npenc: ', t[:,0], t[:,1], t[:,2] )

    #TODO: generalise to `ndi`, currently for `nd`
    # if validate:
    #   t = to_valid_npenc(t)
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

#DONE
def trim_bw_beat(idxenc, pos, vocab, beat_low=None, beat_high=None, include_last_sep=True):
    if beat_high is None or beat_low is None: return idxenc
    cutoff_low = beat2index(idxenc, pos, vocab, beat_low, include_last_sep=include_last_sep)
    # print('cutoff_low: ', cutoff_low)

    cutoff_high = beat2index(idxenc, pos, vocab, beat_high, include_last_sep=include_last_sep)
    # print('cutoff_high: ', cutoff_high)
    return idxenc[cutoff_low:cutoff_high]


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

### **MultiTrack Item**

def multitrack_item_checks(item: MusicItem, first_track, second_track):

  #Check 1: Both instruments should have at least 1 note present in them
  try:
    first_track.to_text().index(' n')
    second_track.to_text().index(' n')
  except:
    print('Critical Error, MultiTrackItem has one of the instruments with no notes!')
    print('First ins: ', first_track.to_text())
    print('Second ins: ', second_track.to_text())
    return False

  #Check 2: There should be at least 1 overlapping note between the two instruments

    return False



  return True


class MultitrackItem():
  def __init__(self, first_instrument:MusicItem, second_instrument:MusicItem, vocab, stream = None):
    self.first_instrument, self.second_instrument = first_instrument, second_instrument 
    self.vocab = vocab
    self._stream = stream
    
    #DONE
    #Represents the lower beat index of the segment that was last given to the S2SPreloader for passing for training
    self.cur_low_beat = 0

    # #If one of the tracks have no notes, raise an exception
    # try:
    #   self.first_instrument.to_text().index(' n')
    #   self.second_instrument.to_text().index(' n')
    # except:
    #   print('Critical Error, MultiTrackItem has one of the instruments with no notes!')
    #   print('First ins: ', self.first_instrument.to_text())
    #   print('Second ins: ', self.second_instrument.to_text())
    #   raise IndexError

    #TODO
    # Add in `from_stream` and `from_npenc` the check that at least one note from both instrument should
    # be present in a common beat, else raise exception 

    # xxsep all indices list -> xxseplst
    # for (a,b) pair in xxseplst:
    #   vocab.stoi[ACCEP_INS['Piano']] and vocab.stoi[ACCEP_INS['Bass']] should be present between a and b indices

  @classmethod
  def from_file(cls, midi_file, vocab, first_ins = 'Piano', second_ins = 'Bass'):
    return cls.from_stream(file2stream(midi_file), vocab, first_ins, second_ins)

  @classmethod
  def from_stream(cls, stream, vocab, first_ins = 'Piano', second_ins = 'Bass'):
    item = MusicItem.from_stream(stream, vocab)

    # Maybe we need to include some check before finding the individual tracks
    # Because the song might noe consist of required instruments 

    first_track = item.to_individual_instrument(ins=first_ins)
    second_track = item.to_individual_instrument(ins=second_ins)

    #Applying checks
    # if not multitrack_item_checks(item, first_track, second_track): raise IndexError

    return MultitrackItem(first_track, second_track, vocab, stream) 

  @classmethod
  def from_npenc(cls, npenc, vocab, stream=None, first_ins = 'Piano', second_ins = 'Bass'):
    item = MusicItem.from_npenc(npenc, vocab)

    first_track = item.to_individual_instrument(ins=first_ins)
    second_track = item.to_individual_instrument(ins=second_ins)

    #Applying checks
    # if not multitrack_item_checks(item, first_track, second_track): raise IndexError

    return MultitrackItem(first_track, second_track, vocab, stream)

  @classmethod
  def from_idx(cls, item, vocab):
      f, s = item
      return MultitrackItem(MusicItem.from_idx(f, vocab), MusicItem.from_idx(s, vocab), vocab)
  
  def to_idx(self): return np.array((self.first_instrument.to_idx(), self.second_instrument.to_idx()))
  
  @property
  def stream(self):
      self._stream = self.to_stream() if self._stream is None else self._stream
      return self._stream
  
  def to_stream(self, bpm=120):
      ps = self.first_instrument.to_npenc(), self.second_instrument.to_npenc()
      # ps = [npenc2chordarr(p) for p in ps]
      chordarr = chordarr_from_multi_npenc(ps)

      # print(chordarr.shape)
      return chordarr2stream(chordarr, bpm=bpm)
      # return npenc2stream(ps, bpm=bpm)

  def show(self, format:str=None):
      return self.stream.show(format)
  def play(self): self.stream.show('midi')
      
  def transpose(self, val):
      return MultitrackItem(self.first_instrument.transpose(val), self.second_instrument.transpose(val), self.vocab)
  def pad_to(self, val):
      return MultitrackItem(self.first_instrument.pad_to(val), self.second_instrument.pad_to(val), self.vocab)
  def trim_to_beat(self, beat):
      return MultitrackItem(self.first_instrument.trim_to_beat(beat), self.second_instrument.trim_to_beat(beat), self.vocab)
  def trim_bw_beat(self, beat_low, beat_high):
      return MultitrackItem(self.first_instrument.trim_bw_beat(beat_low, beat_high), self.second_instrument.trim_bw_beat(beat_low, beat_high), self.vocab)

  #DONE
  def segment_to_parts(self, bptt = 512, beat_delta = 4, SAMPLE_FREQ = 4, verbose = False):
      cur_upper_beat = self.cur_low_beat
      BRK_FLAG = False


      first_ins_seg_cur = self.first_instrument.trim_bw_beat(self.cur_low_beat, cur_upper_beat + beat_delta)
      second_ins_seg_cur = self.second_instrument.trim_bw_beat(self.cur_low_beat, cur_upper_beat + beat_delta)

      first_ins_seg_len = len(first_ins_seg_cur)
      second_ins_seg_len = len(second_ins_seg_cur)
      max_seq_len = max(first_ins_seg_len, second_ins_seg_len)

      if verbose:
        print('cur_upper_beat = self.cur_low_beat = ', cur_upper_beat)
        print('first_ins_seg_len = ', first_ins_seg_len)
        print('second_ins_seg_len = ', second_ins_seg_len)
        print('max_seq_len = ', max_seq_len)

      first_ins_seg_next = None
      second_ins_seg_next = None

      #Iterating in segments of 4 beats until the sequence length is filled 
      while max_seq_len < bptt:
        if first_ins_seg_next is not None and second_ins_seg_next is not None:
          first_ins_seg_cur = first_ins_seg_next
          second_ins_seg_cur = second_ins_seg_next

        cur_upper_beat = cur_upper_beat + beat_delta

        #If we have reached the end of the song, reset the low_beat index and return 
        if cur_upper_beat + beat_delta >= self.first_instrument.position[-1]//SAMPLE_FREQ:
          BRK_FLAG = True
          if verbose:
            print('FLAG SET: ')
            print('cur_upper_beat: ', cur_upper_beat)
            print('self.first_instrument.position[-1]//SAMPLE_FREQ: ', self.first_instrument.position[-1]//SAMPLE_FREQ)
          break
        
        first_ins_seg_next = self.first_instrument.trim_bw_beat(self.cur_low_beat, cur_upper_beat + beat_delta)
        second_ins_seg_next = self.second_instrument.trim_bw_beat(self.cur_low_beat, cur_upper_beat + beat_delta)

        first_ins_seg_len = len(first_ins_seg_next)
        second_ins_seg_len = len(second_ins_seg_next)

        max_seq_len = max(first_ins_seg_len, second_ins_seg_len)
        

      
      #If either of the current segments have no notes, move onto the next segment
      try:
        if verbose:
          print('TRY ->>>')
          print(first_ins_seg_cur.ins)
          print(first_ins_seg_cur.to_text())
          print(second_ins_seg_cur.ins)
          print(second_ins_seg_cur.to_text())

        #If any note is found in both the instruments, pass
        if not first_ins_seg_cur.to_text()[0] == 'n':
          first_ins_seg_cur.to_text().index(' n')
        if not second_ins_seg_cur.to_text()[0] == 'n':
          second_ins_seg_cur.to_text().index(' n')
          
      except:
        if not BRK_FLAG:
          self.cur_low_beat = cur_upper_beat
        else:
          self.cur_low_beat = 0

        if verbose:
          print('Empty segment detected in either instrument!! Moving onto next segment with self.cur_low_beat = ', self.cur_low_beat )
          # print(first_ins_seg.to_text())
          # print(second_ins_seg.to_text())
          print('---------------')
        return self.segment_to_parts(bptt, beat_delta, SAMPLE_FREQ, verbose)
      
      cur_low_beat_copy = self.cur_low_beat

      if verbose:
        print('updated cur_upper_beat = ', cur_upper_beat)
        print('BRK_FLAG = ', BRK_FLAG)

      if not BRK_FLAG:
        self.cur_low_beat = cur_upper_beat
      else:
        self.cur_low_beat = 0

      return self.trim_bw_beat(cur_low_beat_copy, cur_upper_beat)


def chordarr_from_multi_npenc(ps, note_size=NOTE_SIZE):

    num_instruments = len(ACCEP_INS)
    print(ps[0].shape, ps[1].shape)
    print(num_instruments)

    max_len = max(npenc_len(ps[0]), npenc_len(ps[1]))
    # score_arr = (steps, inst, note)
    score_arr = np.zeros((max_len, num_instruments + 1, note_size))
    
    for npenc in ps:
      idx = 0
      for step in npenc:
          n,d,i = (step.tolist()+[0])[:3] # or n,d,i
          if n < VALTSEP: continue # special token
          if n == VALTSEP:
              idx += d
              continue
          score_arr[idx,i,n] = d
    return score_arr

from fastai.basics import *
# Sequence 2 Sequence Translate

class S2SFileProcessor(PreProcessor):
    "`PreProcessor` that opens the filenames and read the texts."
    def process_one(self,item):
        out = np.load(item, allow_pickle=True)
      
        # if out.shape != (2,): return None
        # if not 16 < len(out[0]) < 2048: return None
        # if not 16 < len(out[1]) < 2048: return None
        return out
    
    def process(self, ds:Collection):
        ds.items = [self.process_one(item) for item in ds.items]
        ds.items = [i for i in ds.items if i is not None] # filter out None

class S2SPartsProcessor(PreProcessor):
    "Encodes midi file into 2 separate parts - melody and chords."
    
    def process_one(self, item):
        mtrack = MultitrackItem.from_npenc(item, vocab=self.vocab, first_ins='Piano', second_ins='Bass')
        return mtrack.to_idx()
    
    def process(self, ds):
        self.vocab = ds.vocab
        ds.items = [self.process_one(item) for item in ds.items]
        
class Midi2MultitrackProcessor(PreProcessor):
    "Converts midi files to multitrack items"
    def process_one(self, midi_file):
        try:
            item = MultitrackItem.from_file(midi_file, vocab=self.vocab)
        except Exception as e:
            print(e)
            return None
        return item.to_idx()
        
    def process(self, ds):
        self.vocab = ds.vocab
        ds.items = [self.process_one(item) for item in ds.items]
        ds.items = [i for i in ds.items if i is not None]
    
class S2SPreloader(Callback):
    def __init__(self, dataset:LabelList, bptt:int=512, 
                 transpose_range=None, **kwargs):
        self.dataset,self.bptt = dataset,bptt
        self.vocab = self.dataset.vocab
        self.transpose_range = transpose_range
        # self.rand_transpose = partial(rand_transpose_value, rand_range=transpose_range) if transpose_range is not None else None
        self.rand_transpose = None
        print('S2SPreloader: transpose_range = ', transpose_range)
        
    def __getitem__(self, k:int):
        item,empty_label = self.dataset[k]
        if self.rand_transpose is not None:
            val = self.rand_transpose()
            item = item.transpose(val)
            print('NAAAAAAAAAAAADNIIIIIIIIIIII!???')

        #Automatically functions as CircularIndex to generate different segments of the MIDI when `segment_to_parts` is called
        item = item.segment_to_parts()

        item = item.pad_to(self.bptt+1)
        # melody -> first ins, chords -> second ins 
        ((f_x, f_pos), (s_x, s_pos)) = item.to_idx()
        return f_x, f_pos, s_x, s_pos
    
    def __len__(self):
        return len(self.dataset)

def rand_transpose_value(rand_range=(0,24), p=0.5):
    if np.random.rand() < p: return np.random.randint(*rand_range)-rand_range[1]//2
    return 0

class S2SItemList(MusicItemList):
    _bunch = MusicDataBunch
    def get(self, i):
        return MultitrackItem.from_idx(self.items[i], self.vocab)

# DATALOADING AND TRANSFORMATIONS
# These transforms happen on batch

def mask_tfm(b, mask_range, mask_idx, pad_idx, p=0.3):
    # mask range (min, max)
    # replacement vals - [x_replace, y_replace]. Usually [mask_idx, pad_idx]
    # p = replacement probability
    x,y = b
    x,y = x.clone(),y.clone()
    rand = torch.rand(x.shape, device=x.device)
    rand[x < mask_range[0]] = 1.0
    rand[x >= mask_range[1]] = 1.0
    
    # p(15%) of words are replaced. Of those p(15%) - 80% are masked. 10% wrong word. 10% unchanged
    y[rand > p] = pad_idx # pad unchanged 80%. Remove these from loss/acc metrics
    x[rand <= (p*.8)] = mask_idx # 80% = mask
    wrong_word = (rand > (p*.8)) & (rand <= (p*.9)) # 10% = wrong word
    x[wrong_word] = torch.randint(*mask_range, [wrong_word.sum().item()], device=x.device)
    return x, y

def mask_lm_tfm_default(b, vocab, mask_p=0.3):
    return mask_lm_tfm(b, mask_range=vocab.npenc_range, mask_idx=vocab.mask_idx, pad_idx=vocab.pad_idx, mask_p=mask_p)

def mask_lm_tfm_pitchdur(b, vocab, mask_p=0.9):
    mask_range = vocab.dur_range if np.random.rand() < 0.5 else vocab.note_range
    return mask_lm_tfm(b, mask_range=mask_range, mask_idx=vocab.mask_idx, pad_idx=vocab.pad_idx, mask_p=mask_p)

def mask_lm_tfm(b, mask_range, mask_idx, pad_idx, mask_p):
    x,y = b
    x_lm,x_pos = x[...,0], x[...,1]
    y_lm,y_pos = y[...,0], y[...,1]
    
    # Note: masking y_lm instead of x_lm. Just in case we ever do sequential s2s training
    x_msk, y_msk = mask_tfm((y_lm, y_lm), mask_range=mask_range, mask_idx=mask_idx, pad_idx=pad_idx, p=mask_p)
    msk_pos = y_pos
    
    x_dict = { 
        'msk': { 'x': x_msk, 'pos': msk_pos },
        'lm': { 'x': x_lm, 'pos': msk_pos }
    }
    y_dict = { 'msk': y_msk, 'lm': y_lm }
    return x_dict, y_dict

def melody_chord_tfm(b):
    m,m_pos,c,c_pos = b
    
    # offset x and y for next word prediction
    y_m = m[:,1:]
    x_m, m_pos = m[:,:-1], m_pos[:,:-1]
    
    y_c = c[:,1:]
    x_c, c_pos = c[:,:-1], c_pos[:,:-1]
    
    x_dict = { 
        's2f': {
            'enc': x_c,
            'enc_pos': c_pos,
            'dec': x_m,
            'dec_pos': m_pos
        },
        'f2s': {
            'enc': x_m,
            'enc_pos': m_pos,
            'dec': x_c,
            'dec_pos': c_pos
        }
    }
    y_dict = {
        's2f': y_m, 'f2s': y_c
    }
    return x_dict, y_dict

###**model.py**

import numpy as np
import torch

def window_mask(x_len, device, m_len=0, size=(1,1)):
    win_size,k = size
    mem_mask = torch.zeros((x_len,m_len), device=device)
    tri_mask = torch.triu(torch.ones((x_len//win_size+1,x_len//win_size+1), device=device),diagonal=k)
    window_mask = tri_mask.repeat_interleave(win_size,dim=0).repeat_interleave(win_size,dim=1)[:x_len,:x_len]
    if x_len: window_mask[...,0] = 0 # Always allowing first index to see. Otherwise you'll get NaN loss
    mask = torch.cat((mem_mask, window_mask), dim=1)[None,None]
    return mask.bool() if hasattr(mask, 'bool') else mask.byte()
    
def rand_window_mask(x_len,m_len,device,max_size:int=None,p:float=0.2,is_eval:bool=False):
    if is_eval or np.random.rand() >= p or max_size is None: 
        win_size,k = (1,1)
    else: win_size,k = (np.random.randint(0,max_size)+1,0)
    return window_mask(x_len, device, m_len, size=(win_size,k))

def lm_mask(x_len, device):
    mask = torch.triu(torch.ones((x_len, x_len), device=device), diagonal=1)[None,None]
    return mask.bool() if hasattr(mask, 'bool') else mask.byte()

#### **Music Transformer**

#@title
#https://github.com/bearpelican/musicautobot/blob/master/musicautobot/music_transformer/model.py

from fastai.basics import *
from fastai.text.models.transformer import TransformerXL
#from ..utils.attention_mask import rand_window_mask

class MusicTransformerXL(TransformerXL):
    "Exactly like fastai's TransformerXL, but with more aggressive attention mask: see `rand_window_mask`"
    def __init__(self, *args, encode_position=True, mask_steps=1, **kwargs):
        import inspect
        sig = inspect.signature(TransformerXL)
        arg_params = { k:kwargs[k] for k in sig.parameters if k in kwargs }
        super().__init__(*args, **arg_params)

        self.encode_position = encode_position
        if self.encode_position: self.beat_enc = BeatPositionEncoder(kwargs['d_model'])
            
        self.mask_steps=mask_steps
        
        
    def forward(self, x):
        #The hidden state has to be initiliazed in the forward pass for nn.DataParallel
        if self.mem_len > 0 and not self.init: 
            self.reset()
            self.init = True

        benc = 0
        if self.encode_position:
            # print(x)
            x,pos = x['x'], x['pos']
            benc = self.beat_enc(pos)

        bs,x_len = x.size()
        inp = self.drop_emb(self.encoder(x) + benc) #.mul_(self.d_model ** 0.5)
        m_len = self.hidden[0].size(1) if hasattr(self, 'hidden') and len(self.hidden[0].size()) > 1 else 0
        seq_len = m_len + x_len
        
        mask = rand_window_mask(x_len, m_len, inp.device, max_size=self.mask_steps, is_eval=not self.training) if self.mask else None
        if m_len == 0: mask[...,0,0] = 0
        #[None,:,:None] for einsum implementation of attention
        hids = []
        pos = torch.arange(seq_len-1, -1, -1, device=inp.device, dtype=inp.dtype)
        pos_enc = self.pos_enc(pos)
        hids.append(inp)
        for i, layer in enumerate(self.layers):
            mem = self.hidden[i] if self.mem_len > 0 else None
            inp = layer(inp, r=pos_enc, u=self.u, v=self.v, mask=mask, mem=mem)
            hids.append(inp)
        core_out = inp[:,-x_len:]
        if self.mem_len > 0 : self._update_mems(hids)
        return (self.hidden if self.mem_len > 0 else [core_out]),[core_out]


 # Beat encoder
class BeatPositionEncoder(nn.Module):
    "Embedding + positional encoding + dropout"
    def __init__(self, emb_sz:int, beat_len=32, max_bar_len=1024):
        super().__init__()

        self.beat_len, self.max_bar_len = beat_len, max_bar_len
        self.beat_enc = nn.Embedding(beat_len, emb_sz, padding_idx=0)
        self.bar_enc = nn.Embedding(max_bar_len, emb_sz, padding_idx=0)
    
    def forward(self, pos):
        beat_enc = self.beat_enc(pos % self.beat_len)
        bar_pos = pos // self.beat_len % self.max_bar_len
        bar_pos[bar_pos >= self.max_bar_len] = self.max_bar_len - 1
        bar_enc = self.bar_enc((bar_pos))
        return beat_enc + bar_enc

#### **S2S Transformer**

from fastai.basics import *
from fastai.text.models.transformer import Activation, PositionalEncoding, feed_forward, init_transformer, _line_shift
from fastai.text.models.awd_lstm import RNNDropout

def get_multitask_model(vocab_size:int, config:dict=None, drop_mult:float=1., pad_idx=None):
    "Create a language model from `arch` and its `config`, maybe `pretrained`."
    for k in config.keys(): 
        if k.endswith('_p'): config[k] *= drop_mult
    n_hid = config['d_model']
    mem_len = config.pop('mem_len')
    embed = TransformerEmbedding(vocab_size, n_hid, embed_p=config['embed_p'], mem_len=mem_len, pad_idx=pad_idx)
    encoder = MTEncoder(embed, n_hid, n_layers=config['enc_layers'], mem_len=0, **config) # encoder doesn't need memory
    decoder = MTEncoder(embed, n_hid, is_decoder=True, n_layers=config['dec_layers'], mem_len=mem_len, **config)
    head = MTLinearDecoder(n_hid, vocab_size, tie_encoder=embed.embed, **config)
    model = MultiTransformer(encoder, decoder, head, mem_len=mem_len)
    return model.apply(init_transformer)

class MultiTransformer(nn.Module):
    "Multitask Transformer for training mask, next word, and sequence 2 sequence"
    def __init__(self, encoder, decoder, head, mem_len):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.head = head
        self.default_mem_len = mem_len
        self.current_mem_len = None
    
    def forward(self, inp):
        # data order: mask, next word, first instrument, second instrument
        outputs = {}
        msk, lm, s2f, f2s = [inp.get(key) for key in ['msk', 'lm', 's2f', 'f2s']]
        
        # not necessary for now 
        if msk is not None:
            outputs['msk'] = self.head(self.encoder(msk['x'], msk['pos']))
        if lm is not None:
            outputs['lm'] = self.head(self.decoder(lm['x'], lm['pos']))
        
        if s2f is not None:
            self.reset()
            s2f_enc = self.encoder(s2f['enc'], s2f['enc_pos'])
            s2f_dec = self.decoder(s2f['dec'], s2f['dec_pos'], s2f_enc)
            outputs['s2f'] = self.head(s2f_dec)
            
        if f2s is not None:
            self.reset()
            f2s_enc = self.encoder(f2s['enc'], f2s['enc_pos'])
            f2s_dec = self.decoder(f2s['dec'], f2s['dec_pos'], f2s_enc)
            outputs['f2s'] = self.head(f2s_dec)
            
        return outputs
    
    "A sequential module that passes the reset call to its children."
    def reset(self):
        for module in self.children(): 
            reset_children(module)
        
def reset_children(mod):
    if hasattr(mod, 'reset'): mod.reset()
    for module in mod.children(): 
        reset_children(module)

 # COMPONENTS
class TransformerEmbedding(nn.Module):
    "Embedding + positional encoding + dropout"
    def __init__(self, vocab_size:int, emb_sz:int, embed_p:float=0., mem_len=512, beat_len=32, max_bar_len=1024, pad_idx=None):
        super().__init__()
        self.emb_sz = emb_sz
        self.pad_idx = pad_idx
        
        self.embed = nn.Embedding(vocab_size, emb_sz, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(emb_sz)
        self.beat_len, self.max_bar_len = beat_len, max_bar_len
        self.beat_enc = nn.Embedding(beat_len, emb_sz, padding_idx=0)
        self.bar_enc = nn.Embedding(max_bar_len, emb_sz, padding_idx=0)
        
        self.drop = nn.Dropout(embed_p)
        self.mem_len = mem_len
    
    def forward(self, inp, pos):
        beat_enc = self.beat_enc(pos % self.beat_len)
        bar_pos = pos // self.beat_len % self.max_bar_len
        bar_pos[bar_pos >= self.max_bar_len] = self.max_bar_len - 1
        bar_enc = self.bar_enc((bar_pos))
        emb = self.drop(self.embed(inp) + beat_enc + bar_enc)
        return emb
    
    def relative_pos_enc(self, emb):
#         return torch.arange(640-1, -1, -1).float().cuda()
        seq_len = emb.shape[1] + self.mem_len
        pos = torch.arange(seq_len-1, -1, -1, device=emb.device, dtype=emb.dtype) # backwards (txl pos encoding)
        return self.pos_enc(pos)

class MTLinearDecoder(nn.Module):
    "To go on top of a RNNCore module and create a Language Model."
    initrange=0.1

    def __init__(self, n_hid:int, n_out:int, output_p:float, tie_encoder:nn.Module=None, out_bias:bool=True, **kwargs):
        super().__init__()
        self.decoder = nn.Linear(n_hid, n_out, bias=out_bias)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.output_dp = RNNDropout(output_p)
        if out_bias: self.decoder.bias.data.zero_()
        if tie_encoder: self.decoder.weight = tie_encoder.weight

    def forward(self, input:Tuple[Tensor,Tensor])->Tuple[Tensor,Tensor,Tensor]:
        output = self.output_dp(input)
        decoded = self.decoder(output)
        return decoded

    
# DECODER TRANSLATE BLOCK
class MTEncoder(nn.Module):
    def __init__(self, embed:nn.Module, n_hid:int, n_layers:int, n_heads:int, d_model:int, d_head:int, d_inner:int, 
                 resid_p:float=0., attn_p:float=0., ff_p:float=0., bias:bool=True, scale:bool=True,
                 act:Activation=Activation.ReLU, double_drop:bool=True, mem_len:int=512, is_decoder=False,
                 mask_steps=1, mask_p=0.3, **kwargs):
        super().__init__()
        self.embed = embed
        self.u = nn.Parameter(torch.Tensor(n_heads, 1, d_head)) #Remove 1 for einsum implementation of attention
        self.v = nn.Parameter(torch.Tensor(n_heads, 1, d_head)) #Remove 1 for einsum implementation of attention
        self.n_layers,self.d_model = n_layers,d_model
        self.layers = nn.ModuleList([MTEncoderBlock(n_heads, d_model, d_head, d_inner, resid_p=resid_p, attn_p=attn_p,
                      ff_p=ff_p, bias=bias, scale=scale, act=act, double_drop=double_drop, mem_len=mem_len,
                      ) for k in range(n_layers)])

        self.mask_steps, self.mask_p = mask_steps, mask_p
        self.is_decoder = is_decoder
    
        nn.init.normal_(self.u, 0., 0.02)
        nn.init.normal_(self.v, 0., 0.02)
        
    def forward(self, x_lm, lm_pos, msk_emb=None):
        bs,lm_len = x_lm.size()
        
        lm_emb = self.embed(x_lm, lm_pos)
        if msk_emb is not None and msk_emb.shape[1] > lm_emb.shape[1]:
            pos_enc = self.embed.relative_pos_enc(msk_emb)
        else:
            pos_enc = self.embed.relative_pos_enc(lm_emb)
    
        # Masks
        if self.is_decoder:
            lm_mask = rand_window_mask(lm_len, self.embed.mem_len, x_lm.device,
                                       max_size=self.mask_steps, p=self.mask_p, is_eval=not self.training)
        else:
            lm_mask = None
        
        for i, layer in enumerate(self.layers):
            lm_emb = layer(lm_emb, msk_emb, lm_mask=lm_mask,
                        r=pos_enc, g_u=self.u, g_v=self.v)
        return lm_emb

class MTEncoderBlock(nn.Module):
    "Decoder block of a Transformer model."
    #Can't use Sequential directly cause more than one input...
    def __init__(self, n_heads:int, d_model:int, d_head:int, d_inner:int, resid_p:float=0., attn_p:float=0., ff_p:float=0.,
                 bias:bool=True, scale:bool=True, double_drop:bool=True, mem_len:int=512, mha2_mem_len=0, **kwargs):
        super().__init__()
        attn_cls = MemMultiHeadRelativeAttentionKV
        self.mha1 = attn_cls(n_heads, d_model, d_head, resid_p=resid_p, attn_p=attn_p, bias=bias, scale=scale, mem_len=mem_len, r_mask=False)
        self.mha2 = attn_cls(n_heads, d_model, d_head, resid_p=resid_p, attn_p=attn_p, bias=bias, scale=scale, mem_len=mha2_mem_len, r_mask=True)
        self.ff   = feed_forward(d_model, d_inner, ff_p=ff_p, double_drop=double_drop)
    
    def forward(self, enc_lm:Tensor, enc_msk:Tensor,
                r=None, g_u=None, g_v=None,
                msk_mask:Tensor=None, lm_mask:Tensor=None): 
        
        y_lm = self.mha1(enc_lm, enc_lm, enc_lm, r, g_u, g_v, mask=lm_mask)
        if enc_msk is None: return y_lm
        return self.ff(self.mha2(y_lm, enc_msk, enc_msk, r, g_u, g_v, mask=msk_mask))
    

    # Attention Layer


# Attn

class MemMultiHeadRelativeAttentionKV(nn.Module):
    "Attention Layer monster - relative positioning, keeps track of own memory, separate kv weights to support sequence2sequence decoding."
    def __init__(self, n_heads:int, d_model:int, d_head:int=None, resid_p:float=0., attn_p:float=0., bias:bool=True,
                 scale:bool=True, mem_len:int=512, r_mask=True):
        super().__init__()
        d_head = ifnone(d_head, d_model//n_heads)
        self.n_heads,self.d_head,self.scale = n_heads,d_head,scale
        
        assert(d_model == d_head * n_heads)
        self.q_wgt = nn.Linear(d_model, n_heads * d_head, bias=bias)
        self.k_wgt = nn.Linear(d_model, n_heads * d_head, bias=bias)
        self.v_wgt = nn.Linear(d_model, n_heads * d_head, bias=bias)
        
        self.drop_att,self.drop_res = nn.Dropout(attn_p),nn.Dropout(resid_p)
        self.ln = nn.LayerNorm(d_model)
        self.r_attn = nn.Linear(d_model, n_heads * d_head, bias=bias)
        self.r_mask = r_mask

        self.mem_len = mem_len
        self.prev_k = None
        self.prev_v = None
        
    def forward(self, q:Tensor, k:Tensor=None, v:Tensor=None, 
                r:Tensor=None, g_u:Tensor=None, g_v:Tensor=None, 
                mask:Tensor=None, **kwargs):
        if k is None: k = q
        if v is None: v = q
        return self.ln(q + self.drop_res(self._apply_attention(q, k, v, r, g_u, g_v, mask=mask, **kwargs)))

    def mem_k(self, k):
        if self.mem_len == 0: return k
        if self.prev_k is None or (self.prev_k.shape[0] != k.shape[0]): # reset if wrong batch size
            self.prev_k = k[:, -self.mem_len:]
            return k
        with torch.no_grad():
            k_ext = torch.cat([self.prev_k, k], dim=1)
            self.prev_k = k_ext[:, -self.mem_len:]
        return k_ext.detach()
    
    def mem_v(self, v):
        if self.mem_len == 0: return v
        if self.prev_v is None or (self.prev_v.shape[0] != v.shape[0]): # reset if wrong batch size
            self.prev_v = v[:, -self.mem_len:]
            return v
        with torch.no_grad():
            v_ext = torch.cat([self.prev_v, v], dim=1)
            self.prev_v = v_ext[:, -self.mem_len:]
        return v_ext.detach()
        
    def reset(self):
        self.prev_v = None
        self.prev_k = None
        
    def _apply_attention(self, q:Tensor, k:Tensor, v:Tensor, 
                         r:Tensor=None, g_u:Tensor=None, g_v:Tensor=None, 
                         mask:Tensor=None, **kwargs):
        #Notations from the paper: x input, r vector of relative distance between two elements, u et v learnable
        #parameters of the model common between all layers, mask to avoid cheating and mem the previous hidden states.
#         bs,x_len,seq_len = q.size(0),q.size(1),r.size(0)
        k = self.mem_k(k)
        v = self.mem_v(v)
        bs,x_len,seq_len = q.size(0),q.size(1),k.size(1)
        wq,wk,wv = self.q_wgt(q),self.k_wgt(k),self.v_wgt(v)
        wq = wq[:,-x_len:]
        wq,wk,wv = map(lambda x:x.view(bs, x.size(1), self.n_heads, self.d_head), (wq,wk,wv))
        wq,wk,wv = wq.permute(0, 2, 1, 3),wk.permute(0, 2, 3, 1),wv.permute(0, 2, 1, 3)
        wkr = self.r_attn(r[-seq_len:])
        wkr = wkr.view(seq_len, self.n_heads, self.d_head)
        wkr = wkr.permute(1,2,0)
        #### compute attention score (AC is (a) + (c) and BS is (b) + (d) in the paper)
        AC = torch.matmul(wq+g_u,wk)
        BD = _line_shift(torch.matmul(wq+g_v, wkr), mask=self.r_mask)
        if self.scale: attn_score = (AC + BD).mul_(1/(self.d_head ** 0.5))
        if mask is not None: 
            mask = mask[...,-seq_len:]
            if hasattr(mask, 'bool'): mask = mask.bool()
            attn_score = attn_score.float().masked_fill(mask, -float('inf')).type_as(attn_score)
        attn_prob = self.drop_att(F.softmax(attn_score, dim=-1))
        attn_vec = torch.matmul(attn_prob, wv)
        return attn_vec.permute(0, 2, 1, 3).contiguous().view(bs, x_len, -1)

###**utils**

#@title
#https://github.com/bearpelican/musicautobot/blob/master/musicautobot/utils/top_k_top_p.py

import torch
import torch.nn.functional as F

__all__ = ['top_k_top_p']

# top_k + nucleus filter - https://twitter.com/thom_wolf/status/1124263861727760384?lang=en
# https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
def top_k_top_p(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    logits = logits.clone()
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

#@title
#https://github.com/bearpelican/musicautobot/blob/master/musicautobot/utils/midifile.py

def is_empty_midi(fp):
    if fp is None: return False
    mf = file2mf(fp)
    return not any([t.hasNotes() for t in mf.tracks])

"Parallel processing for midi files"
import csv
from fastprogress.fastprogress import master_bar, progress_bar
from pathlib import Path
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import numpy as np

# https://stackoverflow.com/questions/20991968/asynchronous-multiprocessing-with-a-worker-pool-in-python-how-to-keep-going-aft
def process_all(func, arr, timeout_func=None, total=None, max_workers=None, timeout=None):
    with ProcessPool() as pool:
        future = pool.map(func, arr, timeout=timeout)

        iterator = future.result()
        results = []
        for i in progress_bar(range(len(arr)), total=len(arr)):
            try:
                result = next(iterator)
                if result: results.append(result)
            except StopIteration:
                break  
            except TimeoutError as error:
                if timeout_func: timeout_func(arr[i], error.args[1])
    return results

def process_file(file_path, tfm_func=None, src_path=None, dest_path=None):
    "Utility function that transforms midi file to numpy array."
    output_file = Path(str(file_path).replace(str(src_path), str(dest_path))).with_suffix('.npy')
    if output_file.exists(): return output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Call tfm_func and save file
    npenc = tfm_func(file_path)
    if npenc is not None: 
        np.save(output_file, npenc)
        return output_file

def arr2csv(arr, out_file):
    "Convert metadata array to csv"
    all_keys = {k for d in arr for k in d.keys()}
    arr = [format_values(x) for x in arr]
    with open(out_file, 'w') as f:
        dict_writer = csv.DictWriter(f, list(all_keys))
        dict_writer.writeheader()
        dict_writer.writerows(arr)
        
def format_values(d):
    "Format array values for csv encoding"
    def format_value(v):
        if isinstance(v, list): return ','.join(v)
        return v
    return {k:format_value(v) for k,v in d.items()}

###**learner.py**

#### **Music Transformer**

#@title
#https://github.com/bearpelican/musicautobot/blob/15bc523548f8ae737a594ee92564538d02e0dc94/musicautobot/music_transformer/learner.py

from fastai.basics import *
from fastai.text.learner import LanguageLearner, get_language_model, _model_meta
#from .model import *
#from .transform import MusicItem
#from ..numpy_encode import SAMPLE_FREQ
#from ..utils.top_k_top_p import top_k_top_p
#from ..utils.midifile import is_empty_midi

_model_meta[MusicTransformerXL] = _model_meta[TransformerXL] # copy over fastai's model metadata

def music_model_learner(data:DataBunch, arch=MusicTransformerXL, config:dict=None, drop_mult:float=1.,
                        pretrained_path:PathOrStr=None, encode_position = True, **learn_kwargs) -> 'LanguageLearner':
    "Create a `Learner` with a language model from `data` and `arch`."
    meta = _model_meta[arch]

    if pretrained_path: 
        state = torch.load(pretrained_path, map_location='cpu')
        if config is None: config = state['config']
        
    model = get_language_model(arch, len(data.vocab.itos), config=config, drop_mult=drop_mult)
    #DONE
    # if hasattr(model[0], 'encode_position'):
      # model[0].encode_position = encode_position
    learn = MusicLearner(data, model, split_func=meta['split_lm'], **learn_kwargs)

    if pretrained_path: 
        get_model(model).load_state_dict(state['model'], strict=False)
        if not hasattr(learn, 'opt'): learn.create_opt(defaults.lr, learn.wd)
        try:    learn.opt.load_state_dict(state['opt'])
        except: pass
        del state
        gc.collect()

    return learn

# Predictions
from fastai import basic_train # for predictions
class MusicLearner(LanguageLearner):
    def save(self, file:PathLikeOrBinaryStream=None, with_opt:bool=True, config=None):
        "Save model and optimizer state (if `with_opt`) with `file` to `self.model_dir`. `file` can be file-like (file or buffer)"
        out_path = super().save(file, return_path=True, with_opt=with_opt)
        if config and out_path:
            state = torch.load(out_path)
            state['config'] = config
            torch.save(state, out_path)
            del state
            gc.collect()
        return out_path

    def beam_search(self, xb:Tensor, n_words:int, top_k:int=10, beam_sz:int=10, temperature:float=1.,
                    ):
        "Return the `n_words` that come after `text` using beam search."
        self.model.reset()
        self.model.eval()
        xb_length = xb.shape[-1]
        if xb.shape[0] > 1: xb = xb[0][None]
        yb = torch.ones_like(xb)

        nodes = None
        xb = xb.repeat(top_k, 1)
        nodes = xb.clone()
        scores = xb.new_zeros(1).float()
        with torch.no_grad():
            for k in progress_bar(range(n_words), leave=False):
                out = F.log_softmax(self.model(xb)[0][:,-1], dim=-1)
                values, indices = out.topk(top_k, dim=-1)
                scores = (-values + scores[:,None]).view(-1)
                indices_idx = torch.arange(0,nodes.size(0))[:,None].expand(nodes.size(0), top_k).contiguous().view(-1)
                sort_idx = scores.argsort()[:beam_sz]
                scores = scores[sort_idx]
                nodes = torch.cat([nodes[:,None].expand(nodes.size(0),top_k,nodes.size(1)),
                                indices[:,:,None].expand(nodes.size(0),top_k,1),], dim=2)
                nodes = nodes.view(-1, nodes.size(2))[sort_idx]
                self.model[0].select_hidden(indices_idx[sort_idx])
                xb = nodes[:,-1][:,None]
        if temperature != 1.: scores.div_(temperature)
        node_idx = torch.multinomial(torch.exp(-scores), 1).item()
        return [i.item() for i in nodes[node_idx][xb_length:] ]

    def predict(self, item:MusicItem, n_words:int=128,
                     temperatures:float=(1.0,1.0), min_bars=4,
                     top_k=30, top_p=0.6, allowed_ins:list = None):
        "Return the `n_words` that come after `text`."
        self.model.reset()
        new_idx = []
        vocab = self.data.vocab
        x, pos = item.to_tensor(), item.get_pos_tensor()
        last_pos = pos[-1] if len(pos) else 0
        y = torch.tensor([0])

        start_pos = last_pos

        sep_count = 0
        bar_len = SAMPLE_FREQ * 4 # assuming 4/4 time
        vocab = self.data.vocab

        repeat_count = 0
        if hasattr(self.model[0], 'encode_position'):
            encode_position = self.model[0].encode_position
        else: encode_position = False

        #DONE
        last_xxsep = False

        for i in progress_bar(range(n_words), leave=True):
            with torch.no_grad():
                if encode_position:
                    batch = { 'x': x[None], 'pos': pos[None] }
                    logits = self.model(batch)[0][-1][-1]
                else:
                    logits = self.model(x[None])[0][-1][-1]

            #DONE
            # prev_idx = new_idx[-1] if len(new_idx) else vocab.pad_idx
            prev_idx = new_idx[-1] if len(new_idx) else item.data[-1]
            if prev_idx == vocab.sep_idx:
              last_xxsep = True
            elif vocab.is_ins(prev_idx):
              if prev_idx == vocab.ni_idx:
                last_xxsep = False

            # Temperature
            # Use first temperatures value if last prediction was duration
            temperature = temperatures[0] if vocab.is_duration_or_pad(prev_idx) else temperatures[1]
            repeat_penalty = max(0, np.log((repeat_count+1)/4)/5) * temperature
            temperature += repeat_penalty
            if temperature != 1.: logits = logits / temperature
                

            # Filter
            # bar = 16 beats
            filter_value = -float('Inf')
            if ((last_pos - start_pos) // 16) <= min_bars: logits[vocab.bos_idx] = filter_value

            logits = filter_invalid_indexes(logits, prev_idx, vocab, filter_value=filter_value, last_xxsep = last_xxsep, allowed_ins = allowed_ins)
            logits = top_k_top_p(logits, top_k=top_k, top_p=top_p, filter_value=filter_value)
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1).item()

            # Update repeat count
            num_choices = len(probs.nonzero().view(-1))
            if num_choices <= 2: repeat_count += 1
            else: repeat_count = repeat_count // 2

            if prev_idx==vocab.sep_idx: 
                duration = idx - vocab.dur_range[0]
                last_pos = last_pos + duration

                bars_pred = (last_pos - start_pos) // 16
                abs_bar = last_pos // 16
                # if (bars % 8 == 0) and (bars_pred > min_bars): break
                if (i / n_words > 0.80) and (abs_bar % 4 == 0): break


            if idx==vocab.bos_idx: 
                print('Predicted BOS token. Returning prediction...')
                break

            new_idx.append(idx)
            x = x.new_tensor([idx])
            pos = pos.new_tensor([last_pos])
        #DONE
        #pred = vocab.to_music_item(np.array(new_idx))
        pred = vocab.to_music_item(np.array(new_idx), item.ins)
        full = item.append(pred)
        return pred, full
    
# High level prediction functions from midi file
def predict_from_midi(learn, midi=None, n_words=400, 
                      temperatures=(1.0,1.0), top_k=30, top_p=0.6, seed_len=None, **kwargs):
    vocab = learn.data.vocab
    seed = MusicItem.from_file(midi, vocab) if not is_empty_midi(midi) else MusicItem.empty(vocab)
    if seed_len is not None: seed = seed.trim_to_beat(seed_len)

    pred, full = learn.predict(seed, n_words=n_words, temperatures=temperatures, top_k=top_k, top_p=top_p, **kwargs)
    return full

def filter_invalid_indexes(res, prev_idx, vocab, filter_value=-float('Inf'), last_xxsep = False, allowed_ins:list = None):
    global ACCEP_INS
    #REMOVE
    # print('filter_invalid_indexes called')
    
    #DONE : Hardcoded piano to not be generated at all
    #res[[vocab.ins_range[0]]] = filter_value

    #DONE : Hardcoded every instrument other than violin to not be generated at all
    if allowed_ins is not None:
      res[ list( set(range(vocab.ins_range[0], vocab.ins_range[1])) - set([vocab.stoi['i' + str(ACCEP_INS[x])] for x in allowed_ins]) ) ] = filter_value


    #If the last predicted note was xxsep, then it should be impossible to predict instrument other than xxni
    if last_xxsep is True:
      res[list(range(*vocab.ins_range))] = filter_value
    else:
      res[[vocab.stoi[IN]]] = filter_value

    if vocab.is_duration_or_pad(prev_idx):
        res[list(range(*vocab.dur_range))] = filter_value
        #DONE
        res[list(range(*vocab.note_range))] = filter_value
        res[ list( set([vocab.stoi[x] for x in SPECIAL_TOKS]) - {vocab.stoi[IN]} ) ] = filter_value

    #DONE
    elif vocab.is_ins(prev_idx):
        res[list(range(*vocab.ins_range))] = filter_value
        # res[[vocab.ni_idx]] = filter_value
        res[list(range(*vocab.dur_range))] = filter_value
        res[ list( set([vocab.stoi[x] for x in SPECIAL_TOKS]) - {vocab.stoi[SEP]} ) ] = filter_value   

    else:
        res[list(range(*vocab.note_range))] = filter_value
        #DONE
        res[list(range(*vocab.ins_range))] = filter_value
        res[ list( set([vocab.stoi[x] for x in SPECIAL_TOKS])) ] = filter_value   
        # res[[vocab.ni_idx]] = filter_value
    return res

#### **S2S Transformer**

from fastai.basics import *

def multitask_model_learner(data:DataBunch, config:dict=None, drop_mult:float=1., 
                            pretrained_path:PathOrStr=None, **learn_kwargs) -> 'LanguageLearner':
    "Create a `Learner` with a language model from `data` and `arch`."
    vocab = data.vocab
    vocab_size = len(vocab)

    if pretrained_path: 
        state = torch.load(pretrained_path, map_location='cpu')
        if config is None: config = state['config']

    model = get_multitask_model(vocab_size, config=config, drop_mult=drop_mult, pad_idx=vocab.pad_idx)
    #DONE: Changed metric functions and restricted only to f2s and s2f seq-to-seq tasks
    # metrics = [AverageMultiMetric(partial(m, pad_idx=vocab.pad_idx)) for m in [mask_acc, lm_acc, c2m_acc, m2c_acc]]
    metrics = [AverageMultiMetric(partial(m, pad_idx=vocab.pad_idx)) for m in [s2f_acc, f2s_acc]]
    loss_func = MultiLoss(ignore_index=data.vocab.pad_idx)
    learn = MultitaskLearner(data, model, loss_func=loss_func, metrics=metrics, **learn_kwargs)
    
    if pretrained_path: 
        get_model(model).load_state_dict(state['model'], strict=False)
        if not hasattr(learn, 'opt'): learn.create_opt(defaults.lr, learn.wd)
        try:    learn.opt.load_state_dict(state['opt'])
        except: pass
        del state
        gc.collect()
        
    return learn

class MultitaskLearner(Learner):
    def save(self, file:PathLikeOrBinaryStream=None, with_opt:bool=True, config=None):
        "Save model and optimizer state (if `with_opt`) with `file` to `self.model_dir`. `file` can be file-like (file or buffer)"
        out_path = super().save(file, return_path=True, with_opt=with_opt)
        if config and out_path:
            state = torch.load(out_path)
            state['config'] = config
            torch.save(state, out_path)
            del state
            gc.collect()
        return out_path

    def predict_nw(self, item:MusicItem, n_words:int=128,
                     temperatures:float=(1.0,1.0), min_bars=4,
                     top_k=30, top_p=0.6):
        "Return the `n_words` that come after `text`."
        self.model.reset()
        new_idx = []
        vocab = self.data.vocab
        x, pos = item.to_tensor(), item.get_pos_tensor()
        last_pos = pos[-1] if len(pos) else 0
        y = torch.tensor([0])

        start_pos = last_pos

        sep_count = 0
        bar_len = SAMPLE_FREQ * 4 # assuming 4/4 time
        vocab = self.data.vocab

        repeat_count = 0

        for i in progress_bar(range(n_words), leave=True):
            batch = { 'lm': { 'x': x[None], 'pos': pos[None] } }, y
            logits = self.pred_batch(batch=batch)['lm'][-1][-1]

            prev_idx = new_idx[-1] if len(new_idx) else vocab.pad_idx

            # Temperature
            # Use first temperatures value if last prediction was duration
            temperature = temperatures[0] if vocab.is_duration_or_pad(prev_idx) else temperatures[1]
            repeat_penalty = max(0, np.log((repeat_count+1)/4)/5) * temperature
            temperature += repeat_penalty
            if temperature != 1.: logits = logits / temperature
                

            # Filter
            # bar = 16 beats
            filter_value = -float('Inf')
            if ((last_pos - start_pos) // 16) <= min_bars: logits[vocab.bos_idx] = filter_value

            logits = filter_invalid_indexes(logits, prev_idx, vocab, filter_value=filter_value)
            logits = top_k_top_p(logits, top_k=top_k, top_p=top_p, filter_value=filter_value)
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1).item()

            # Update repeat count
            num_choices = len(probs.nonzero().view(-1))
            if num_choices <= 2: repeat_count += 1
            else: repeat_count = repeat_count // 2

            if prev_idx==vocab.sep_idx: 
                duration = idx - vocab.dur_range[0]
                last_pos = last_pos + duration

                bars_pred = (last_pos - start_pos) // 16
                abs_bar = last_pos // 16
                # if (bars % 8 == 0) and (bars_pred > min_bars): break
                if (i / n_words > 0.80) and (abs_bar % 4 == 0): break


            if idx==vocab.bos_idx: 
                print('Predicted BOS token. Returning prediction...')
                break

            new_idx.append(idx)
            x = x.new_tensor([idx])
            pos = pos.new_tensor([last_pos])

        pred = vocab.to_music_item(np.array(new_idx))
        full = item.append(pred)
        return pred, full

    def predict_mask(self, masked_item:MusicItem,
                    temperatures:float=(1.0,1.0),
                    top_k=20, top_p=0.8):
        x = masked_item.to_tensor()
        pos = masked_item.get_pos_tensor()
        y = torch.tensor([0])
        vocab = self.data.vocab
        self.model.reset()
        mask_idxs = (x == vocab.mask_idx).nonzero().view(-1)

        repeat_count = 0

        for midx in progress_bar(mask_idxs, leave=True):
            prev_idx = x[midx-1]

            # Using original positions, otherwise model gets too off track
            # pos = torch.tensor(-position_enc(xb[0].cpu().numpy()), device=xb.device)[None]
    
            # Next Word
            logits = self.pred_batch(batch=({ 'msk': { 'x': x[None], 'pos': pos[None] } }, y) )['msk'][0][midx]

            # Temperature
            # Use first temperatures value if last prediction was duration
            temperature = temperatures[0] if vocab.is_duration_or_pad(prev_idx) else temperatures[1]
            repeat_penalty = max(0, np.log((repeat_count+1)/4)/5) * temperature
            temperature += repeat_penalty
            if temperature != 1.: logits = logits / temperature

            # Filter
            filter_value = -float('Inf')
            special_idxs = [vocab.bos_idx, vocab.sep_idx, vocab.stoi[EOS]]
            logits[special_idxs] = filter_value # Don't allow any special tokens (as we are only removing notes and durations)
            logits = filter_invalid_indexes(logits, prev_idx, vocab, filter_value=filter_value)
            logits = top_k_top_p(logits, top_k=top_k, top_p=top_p, filter_value=filter_value)

            # Sampling
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1).item()

            # Update repeat count
            num_choices = len(probs.nonzero().view(-1))
            if num_choices <= 2: repeat_count += 1
            else: repeat_count = repeat_count // 2

            x[midx] = idx

        return vocab.to_music_item(x.cpu().numpy())

    def predict_s2s(self, input_item:MusicItem, target_item:MusicItem, n_words:int=256,
                        temperatures:float=(1.0,1.0), top_k=30, top_p=0.8,
                        use_memory=True):
        vocab = self.data.vocab

        #DONE
        target_ins = target_item.ins
        last_xxsep = False 

        # Input doesn't change. We can reuse the encoder output on each prediction
        with torch.no_grad():
            inp, inp_pos = input_item.to_tensor(), input_item.get_pos_tensor()
            x_enc = self.model.encoder(inp[None], inp_pos[None])
        
        # target
        targ = target_item.data.tolist()
        targ_pos = target_item.position.tolist()
        last_pos = targ_pos[-1]
        self.model.reset()

        repeat_count = 0

        max_pos = input_item.position[-1] + SAMPLE_FREQ * 4 # Only predict until both tracks/parts have the same length
        x, pos = inp.new_tensor(targ), inp_pos.new_tensor(targ_pos)
        
        for i in progress_bar(range(n_words), leave=True):
            # Predict
            with torch.no_grad():
                dec = self.model.decoder(x[None], pos[None], x_enc)
                logits = self.model.head(dec)[-1, -1]

            # Temperature
            # Use first temperatures value if last prediction was duration
            prev_idx = targ[-1] if len(targ) else vocab.pad_idx

            #DONE
            if prev_idx == vocab.sep_idx:
              last_xxsep = True
            elif vocab.is_ins(prev_idx):
              if prev_idx == vocab.ni_idx:
                last_xxsep = False

            temperature = temperatures[0] if vocab.is_duration_or_pad(prev_idx) else temperatures[1]
            repeat_penalty = max(0, np.log((repeat_count+1)/4)/5) * temperature
            temperature += repeat_penalty
            if temperature != 1.: logits = logits / temperature
                
            # Filter
            filter_value = -float('Inf')
            logits = filter_invalid_indexes(logits, prev_idx, vocab, filter_value=filter_value, last_xxsep = last_xxsep, allowed_ins = [target_ins])
            logits = top_k_top_p(logits, top_k=top_k, top_p=top_p, filter_value=filter_value)

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1).item()

            # Update repeat count
            num_choices = len(probs.nonzero().view(-1))
            if num_choices <= 2: repeat_count += 1
            else: repeat_count = repeat_count // 2

            if idx == vocab.bos_idx | idx == vocab.stoi[EOS]: 
                print('Predicting BOS/EOS')
                break

            if prev_idx == vocab.sep_idx: 
                duration = idx - vocab.dur_range[0]
                last_pos = last_pos + duration
                if last_pos > max_pos:
                    print('Predicted past counter-part length. Returning early')
                    break

            targ_pos.append(last_pos)
            targ.append(idx)
            
            if use_memory:
                # Relying on memory for kv. Only need last prediction index
                x, pos = inp.new_tensor([targ[-1]]), inp_pos.new_tensor([targ_pos[-1]])
            else:
                # Reset memory after each prediction, since we feeding the whole sequence every time
                self.model.reset()
                x, pos = inp.new_tensor(targ), inp_pos.new_tensor(targ_pos)

        return vocab.to_music_item(np.array(targ))
    
# High level prediction functions from midi file
def nw_predict_from_midi(learn, midi=None, n_words=400, 
                      temperatures=(1.0,1.0), top_k=30, top_p=0.6, seed_len=None, **kwargs):
    vocab = learn.data.vocab
    seed = MusicItem.from_file(midi, vocab) if not is_empty_midi(midi) else MusicItem.empty(vocab)
    if seed_len is not None: seed = seed.trim_to_beat(seed_len)
        
    pred, full = learn.predict_nw(seed, n_words=n_words, temperatures=temperatures, top_k=top_k, top_p=top_p, **kwargs)
    return full

def s2s_predict_from_midi(learn, midi=None, n_words=200, 
                      temperatures=(1.0,1.0), top_k=24, top_p=0.7, seed_len=None, pred_melody=True, **kwargs):
    multitrack_item = MultitrackItem.from_file(midi, learn.data.vocab)
    first_ins, second_ins= multitrack_item.first_instrument, multitrack_item.second_instrument
    inp, targ = (first_ins, second_ins) if pred_melody else (second_ins, first_ins)
    
    # if seed_len is passed, cutoff sequence so we can predict the rest
    if seed_len is not None: targ = targ.trim_to_beat(seed_len)
    targ = targ.remove_eos()
        
    pred = learn.predict_s2s(inp, targ, n_words=n_words, temperatures=temperatures, top_k=top_k, top_p=top_p, **kwargs)
    
    part_order = (pred, inp) if pred_melody else (inp, pred)
    return MultitrackItem(*part_order)

def mask_predict_from_midi(learn, midi=None, predict_notes=True,
                           temperatures=(1.0,1.0), top_k=30, top_p=0.7, section=None, **kwargs):
    item = MusicItem.from_file(midi, learn.data.vocab)
    masked_item = item.mask_pitch(section) if predict_notes else item.mask_duration(section)
    pred = learn.predict_mask(masked_item, temperatures=temperatures, top_k=top_k, top_p=top_p, **kwargs)
    return pred

# LOSS AND METRICS

class MultiLoss():
    def __init__(self, ignore_index=None):
        "Loss mult - Mask, NextWord, Seq2Seq"
        self.loss = CrossEntropyFlat(ignore_index=ignore_index)
        
    def __call__(self, inputs:Dict[str,Tensor], targets:Dict[str,Tensor])->Rank0Tensor:
        losses = [self.loss(inputs[key], target) for key,target in targets.items()]
        return sum(losses)
    
def acc_ignore_pad(input:Tensor, targ:Tensor, pad_idx)->Rank0Tensor:
    if input is None or targ is None: return None
    n = targ.shape[0]
    input = input.argmax(dim=-1).view(n,-1)
    targ = targ.view(n,-1)
    mask = targ != pad_idx
    return (input[mask]==targ[mask]).float().mean()

def acc_index(inputs, targets, key, pad_idx):
    return acc_ignore_pad(inputs.get(key), targets.get(key), pad_idx)
    
def mask_acc(inputs, targets, pad_idx): return acc_index(inputs, targets, 'msk', pad_idx)
def lm_acc(inputs, targets, pad_idx): return acc_index(inputs, targets, 'lm', pad_idx)
def c2m_acc(inputs, targets, pad_idx): return acc_index(inputs, targets, 'c2m', pad_idx)
def m2c_acc(inputs, targets, pad_idx): return acc_index(inputs, targets, 'm2c', pad_idx)

def s2f_acc(inputs, targets, pad_idx): return acc_index(inputs, targets, 's2f', pad_idx)
def f2s_acc(inputs, targets, pad_idx): return acc_index(inputs, targets, 'f2s', pad_idx)


class AverageMultiMetric(AverageMetric):
    "Updated fastai.AverageMetric to support multi task metrics."
    def on_batch_end(self, last_output, last_target, **kwargs):
        "Update metric computation with `last_output` and `last_target`."
        if not is_listy(last_target): last_target=[last_target]
        val = self.func(last_output, *last_target)
        if val is None: return
        self.count += first_el(last_target).size(0)
        if self.world:
            val = val.clone()
            dist.all_reduce(val, op=dist.ReduceOp.SUM)
            val /= self.world
        self.val += first_el(last_target).size(0) * val.detach().cpu()

    def on_epoch_end(self, last_metrics, **kwargs):
        "Set the final result in `last_metrics`."
        if self.count == 0: return add_metrics(last_metrics, 0)
        return add_metrics(last_metrics, self.val/self.count)
    

# MODEL LOADING
class MTTrainer(LearnerCallback):
    "`Callback` that regroups lr adjustment to seq_len, AR and TAR."
    def __init__(self, learn:Learner, dataloaders=None, starting_mask_window=1):
        super().__init__(learn)
        self.count = 1
        self.mw_start = starting_mask_window
        self.dataloaders = dataloaders

    def on_epoch_begin(self, **kwargs):
        "Reset the hidden state of the model."
        model = get_model(self.learn.model)
        model.reset()
        model.encoder.mask_steps = max(self.count+self.mw_start, 100)
        
    def on_epoch_end(self, last_metrics, **kwargs):
        "Finish the computation and sends the result to the Recorder."
        if self.dataloaders is not None: 
            self.learn.data = self.dataloaders[self.count % len(self.dataloaders)]
        self.count += 1
