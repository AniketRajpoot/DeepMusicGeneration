from enum import Enum
import music21
import numpy as np

PIANO_TYPES = list(range(24)) + list(range(80, 96)) # Piano, Synths
PLUCK_TYPES = list(range(24, 40)) + list(range(104, 112)) # Guitar, Bass, Ethnic
BRIGHT_TYPES = list(range(40, 56)) + list(range(56, 80))

PIANO_RANGE = (21, 109) # https://en.wikipedia.org/wiki/Scientific_pitch_notation
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

# Genre Tokens 
ELECTRONIC = 'xxelec'
FOLK = 'xxfolk'
FUNK = 'xxfunk'
JAZZ = 'xxjazz'
POP = 'xxpop'
ROCK = 'xxrock'

# Instrument to be accepted 
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

SPECIAL_TOKS = [BOS, PAD, EOS, MASK, ELECTRONIC, FOLK, FUNK, JAZZ, POP, ROCK, IN, SEP] # Important: SEP token must be last

############## Music 21 Utils #################
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


############## Encoding Utils #################
def file2stream(fp):
    if isinstance(fp, music21.midi.MidiFile): return music21.midi.translate.midiFileToStream(fp)
    return music21.converter.parse(fp)

def npenc2stream(arr,bpm=120, instr_list = None):
    "Converts numpy encoding to music21 stream"
    chordarr = npenc2chordarr(np.array(arr)) # 1.
    return chordarr2stream(chordarr,bpm=bpm, instr_list = instr_list) # 2.

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
    
    ins = dict()

    for idx,part in enumerate(s.parts):
        
        notes=[]
        iterate = False
        
        for elem in part.flat:

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

            elif isinstance(elem,music21.instrument.Instrument) and (elem.instrumentName is  None):
                ins[idx] = 'Misc'
                iterate = True 
            
            if isinstance(elem, music21.note.Note):
                notes.append(note_data(elem.pitch, elem))
            if isinstance(elem, music21.chord.Chord):
                for p in elem.pitches:
                    notes.append(note_data(p, elem)) 

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
    


############## Decoding Utils #################
def npenc2chordarr(npenc,note_size=NOTE_SIZE):
    num_instruments = 1 if npenc.shape[1] <= 2 else npenc.max(axis=0)[-1]
    max_len = npenc_len(npenc)
    # score_arr = (steps, inst, note)
    score_arr = np.zeros((max_len, num_instruments + 1, note_size))
    
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

def partarr2stream(partarr,inst,duration):
    "convert instrument part to music21 chords"

    l = len(ACCEP_INS_REV) 
    inst = inst%l
    part = music21.stream.Part()
    
    if(ACCEP_INS_REV[inst] == 'Piano'):
        part.append(music21.instrument.Piano())
    elif(ACCEP_INS_REV[inst] == 'Bass'):
        part.append(music21.instrument.AcousticBass())
    elif(ACCEP_INS_REV[inst] == 'Guitar'):
        part.append(music21.instrument.AcousticGuitar())  
    elif(ACCEP_INS_REV[inst] == 'WoodwindInstrument'):
        part.append(music21.instrument.TenorSaxophone())
    elif(ACCEP_INS_REV[inst] == 'BrassInstrument'):
        part.append(music21.instrument.Trumpet()) 
    elif(ACCEP_INS_REV[inst] == 'StringInstrument'):
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
    chordarr_comb = np.concatenate(parts_padded, axis=1)
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