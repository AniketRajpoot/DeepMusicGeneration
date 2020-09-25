# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 00:08:49 2020

@author: beniw
"""


""" This module prepares midi file data and feeds it to the neural
    network for training """
    
import os
import glob
import pickle
import numpy as np
from music21 import *
from collections import Counter 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout, Masking
from keras.layers import LSTM, Bidirectional
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras_self_attention import SeqSelfAttention
from keras.preprocessing.sequence import pad_sequences

file = "data_multi/multi/Autumn-violin-and-piano.mid"
path = 'data'


def get_notes(file):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes_instrument = []
    midi = converter.parse(file)
    print("Parsing %s" % file)

    notes_to_parse = None

    try: # file has instrument parts
        instruments = instrument.partitionByInstrument(midi)
        print(instruments)
        #notes_to_parse = s2.parts[0].recurse()
    except: # file has notes in a flat structure
         notes_to_parse = midi.flat.notesAndRests
 
    
    for part in instruments.parts:
        instrument_name = part.getInstrument(returnDefault=False).instrumentName 
        if(instrument_name != 'Fretless Bass' and instrument_name != 'Piano' and
           instrument_name != 'Acoustic Bass' and instrument_name != 'Electric Guitar'
           and instrument_name != 'Trumpet' and instrument_name != 'Horn' and
           instrument_name != 'Electric Bass' and instrument_name != 'Violin'):
                continue
    #part.makeRests(fillGaps=True)
    #part.quantize(inPlace = True)
    #part = part.allPlayingWhileSounding(instruments.parts, elStream=None)
        instrument_name = part.getInstrument(returnDefault=False).instrumentName 
        print(instrument_name)
        notes_to_parse = part.recurse()
        print('offset : ', part.offset, ",",  notes_to_parse.offset, ' duration : ', part.duration)
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                el = {'t' : element.offset, 'note/chor/rest' :str(element.pitch),
                      'duration' : float(element.quarterLength),
                      'velocity' : element.volume.velocity,'instrument_name' : instrument_name}
                notes_instrument.append(el)
                # if element.offset < 900:
                    #     print(str(element.pitch),' note : ',element.offset, ' duration : ', element.duration)
            elif isinstance(element, chord.Chord):
                el = {'t' : element.offset, 'note/chor/rest' : '.'.join(str(n) for n in element.normalOrder),
                      'duration' : float(element.quarterLength),
                      'velocity' : element.volume.velocity,'instrument_name' : instrument_name}
                notes_instrument.append(el)
            # notes_instrument.append('.'.join(str(n) for n in element.normalOrder))
            # if element.offset < 50:
            #     print(element.normalOrder,' chord : ',element.offset)
            elif isinstance(element, note.Rest): #ADDED
                el = {'t' : element.offset, 'note/chor/rest' :str(element.name),
                      'duration' : float(element.quarterLength),
                      'velocity' : 0, 'instrument_name' : instrument_name}
                notes_instrument.append(el)
            # notes_instrument.append(element.name) #ADDED
            # if element.offset < 17:
            #     print(element.name, element.offset, ' duration : ', element.duration)
    
        
    
    return notes_instrument

# , num_of_patterns = 640
def prepare_sequences(notes_chords, n_vocab, quarter_length = '0', velocity_press = '0', time_previous = '0'):
    """ Prepare the sequences used by the Neural Network """
    num_of_patterns = 2000
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes_chords))
      # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    notes_input = []
    notes_output = []
    quarter_length_in = []
    quarter_length_out = []
    velocity_press_in = []
    velocity_press_out = []
    time_previous_in = []
    time_previous_out = []
    
    for i in range(0, len(notes_chords) - sequence_length, 1):
        notes_in = notes_chords[i:i + sequence_length]
        notes_out = notes_chords[i + sequence_length]
        notes_input.append([note_to_int[char] for char in notes_in])
        notes_output.append(note_to_int[notes_out])

    n_patterns = len(notes_input)
    
    
    notes_normalized_input = np.reshape(notes_input, (n_patterns, sequence_length, 1))
    # quarter_length_in = numpy.reshape(quarter_length_in, (n_patterns, sequence_length, 1))
    # velocity_press_in = numpy.reshape(velocity_press_in, (n_patterns, sequence_length, 1))
    # time_previous_in = numpy.reshape(time_previous_in, (n_patterns, sequence_length, 1))
    
    notes_normalized_input =  notes_normalized_input / float(n_vocab)
    # quarter_normalized_input =  quarter_length_in / float(n_vocab)
    # velocity_normalized_input =  velocity_press_in / float(n_vocab)
    # time_normalized_input =  time_previous_in / float(n_vocab)
    
    notes_output = np_utils.to_categorical(notes_output)
    # quarter_length_out = np_utils.to_categorical(quarter_length_out)
    # velocity_press_out = np_utils.to_categorical(velocity_press_out)
    # time_previous_out = np_utils.to_categorical(time_previous_out)


    return (notes_input, notes_normalized_input, notes_output)

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    
    # WITHOUT ATTENTION ------> 3 LSTM Layer + 3 Dropout Layers + 2 Dense Layers
    
    # model = Sequential()
    # model.add(LSTM(
    #     512,
    #     input_shape=(network_input.shape[1], network_input.shape[2]),
    #     return_sequences=True
    # ))
    # model.add(Dropout(0.3))
    # model.add(LSTM(512, return_sequences=True))
    # model.add(Dropout(0.3))
    # model.add(LSTM(512))
    # model.add(Dense(256))
    # model.add(Dropout(0.3))
    # model.add(Dense(n_vocab))
    # model.add(Activation('softmax'))
    
    # WITH ATTENTION --------> 1 BiDirectional LSTM Layer + 1 Attention Layer + 1 LSTM Layer
    
    model = Sequential()
    model.add(Bidirectional(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    )))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(LSTM(512))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Dropout(0.2))
    #model.add(Flatten()) #Supposedly needed to fix stuff before dense layer
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    
    
    # opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer= 'rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train(model, network_input, network_output):
    """ train the neural network """
    
    es = EarlyStopping(monitor='loss', patience=35)
    filepath = "weights/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint, es]

    model.fit(network_input, network_output, epochs=50, batch_size = 64, callbacks=callbacks_list)
    
def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    start = np.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    #output_notes = []
    output_notes = stream.Score()
    part_piano = '0'
    part_violin = '0'
    
    # create note and chord objects based on the values generated by the model
    # for pattern in prediction_output:
    for pattern in notes_chords:
        pattern = pattern.split('-')
        print(pattern)
        temp = pattern[0]
        #dur = pattern[1]
        #tsp = pattern[2]
        inst_name = pattern[1]
        pattern = temp
        #offset += convert_to_float(tsp)
        #dur = convert_to_float(dur)
        
        if inst_name == 'Piano':
            sound = instrument.Piano()
            if part_piano == '0':
                part_piano = stream.Part()
            part =  part_piano
            part.insert(sound)
        elif inst_name == 'Acoustic Bass':
            sound = instrument.AcousticBass()
        elif inst_name == 'Electric Guitar':
            sound = instrument.ElectricGuitar()
        elif inst_name == 'Trumpet':
            sound = instrument.Trumpet()
        elif inst_name == 'Horn':
            sound = instrument.Horn()
        elif inst_name == 'Violin':
            sound = instrument.Violin()
            if part_violin == '0':
                part_violin = stream.Part()
            part =  part_violin
            part.insert(sound)
            
        # pattern is a chord   
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            new_meas_note = stream.Measure()
            for current_note in notes_in_chord:
                #new_note = note.Note(int(current_note))
                #new_note.storedInstrument = sound
                #notes.append(new_note)
                new_note = note.Note(int(current_note))
                #new_note.offset = offset
                new_meas_note.append(new_note)
                
            new_meas_chord = stream.Measure()
            new_chord = chord.Chord(new_meas_note)
            new_chord.offset = offset
            new_meas.append(new_note)
            
            #new_chord = chord.Chord(notes)
            #new_chord.quarterLength = dur
            #new_chord.offset = offset
            #output_notes.append(new_chord)
            #part.insert(sound)
            part.append(new_meas_chord)
        # pattern is a rest
        elif('rest' in pattern):
            new_meas = stream.Measure()
            new_rest = note.Rest(pattern)
            new_rest.offset = offset
            new_meas.append(new_note)
            #new_rest.quarterLength = dur
            #new_rest.storedInstrument = sound #???
            #output_notes.append(new_rest)
            #part.insert(sound)
            part.append(new_meas)
        # pattern is a note
        else:
            new_meas = stream.Measure()
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_meas.append(new_note)
            #new_note.quarterLength = dur
            #new_note.offset = offset
            #new_note.storedInstrument = sound
            #output_notes.append(new_note)
            #part.insert(sound)
            part.append(new_meas)
        # increase offset each iteration so that notes do not stack
        # offset += convert_to_float(duration)
        offset += 0.5

    output_notes.insert(0,part_piano)
    output_notes.insert(0,part_violin)
    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='test_output.mid')
 
#From: https://stackoverflow.com/questions/1806278/convert-fraction-to-float
def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac
    
    
# GET THE NOTES READY FOR TRAINING

# Getting the dictionary ----> time, note, duration, instrumentName
        
def main():
    files = [i for i in os.listdir(path) if i.endswith(".mid")] 
    notes = []
    for i in range(len(files)):
        temp = get_notes(os.path.join(path,files[0]))
        notes.extend(temp)
    notes = sorted(notes, key=lambda k: (k['t']))
    
    
    # concatenated note ------> note/chord/rest, duration, time_since_previous, instrument name
    notes_chords = []
    quarter_length = []
    time_previous = []
    velocity_press = []
    for i in range(len(notes)):
        el = notes[i]
        if el['t'] == 0:
            time_since_previous = 0
        else:
            time_since_previous = el['t'] - notes[i-1]['t']
        notes_chords.append(str(el['note/chor/rest']) + '-' + str(el['instrument_name']))
        time_previous.append(time_since_previous)
        quarter_length.append(el['duration'])
        velocity_press.append(el['velocity'])
    
    # get amount of pitch names
    # flatten_notes = []
    # for element in notes:
    #     for note in element:
    #         flatten_notes.append(note)
    
    
    unique_notes = list(set(notes_chords))
    n_vocab = len(set(notes_chords))
    print("Unique Notes : ",n_vocab)    
        
    #freq = dict(Counter(notes))
        
    # no = [count for _,count in freq.items()]
    # plt.figure(figsize=(5,5))
    # plt.hist(no)    
        
    # #from the plot tweak the threshold 
    # threshold = 5
        
    # frequent_notes = [note_ for note_, count in freq.items() if count>=threshold]
    # print("Frequent Notes : ",len(frequent_notes))
        
    #storing the most frequent notes 
    # final_notes = []
                    
    # for note in notes:
    #     if note in frequent_notes:
    #         final_notes.append(note)
                    
    # x_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_notes))
    # x_int_to_note = dict((number, note_) for number, note_ in enumerate(unique_notes)) 
    
        
    # vectorized_list = numpy.array([x_note_to_int[char] for char in final_notes])
        
    # notes = notes[:30000]
    #notes = notes[30000:60000]
    # n_vocab = len(set(notes))
    # print("Updated Unique Notes : ",n_vocab)
        
#    with open('data/notes', 'wb') as filepath:
#        pickle.dump(notes_chords, filepath)
    
    # PREPARING SEQUENCES FOR TRAINING
    network_input, normalized_input, network_output = prepare_sequences(notes_chords,n_vocab)
                                                                        # quarter_length,
                                                                        # velocity_press,
                                                                        # time_previous)
    # flatten_notes = (numpy.array(normalized_input)).flatten()
    # n_vocab = len(set(list(flatten_notes)))

    
    
    # DEFINING THE MODEL ----> GIVING IT A STRUCTURE
    model = create_network(normalized_input, n_vocab)
    
    # TRAINING
    train(model, normalized_input, network_output)
     
        
    # PREDICT
        
#    with open('data/notes', 'rb') as filepath:
#            notes = pickle.load(filepath)
    
    #import random
    
    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # Get all pitch names
    n_vocab = len(set(notes))
    #random.shuffle(network_input)
    #new_model = create_network(normalized_input, n_vocab)
    #new_model.load_weights('weights-improvement-04-4.6496-bigger.hdf5')
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)
    create_midi(notes_chords)

    
if __name__ == "__main__":
    main()


