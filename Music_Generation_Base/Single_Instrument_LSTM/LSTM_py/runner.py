import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import Counter 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import mitdeeplearning as mdl
import random
from tensorflow import keras
from tensorflow.keras import layers
import music21
import midifile 
from enum import Enum
from keras import models
import midifile 
import pre_process

#specifying data paths 
path = 'sample_data'

#tokenizing
BOS = 'xxbos'
PAD = 'xxpad'
EOS = 'xxeos'
MASK = 'xxmask' 
CSEQ = 'xxcseq'
MSEQ = 'xxmseq'
S2SCLS = 'xxs2scls' # deprecated
NSCLS = 'xxnscls' # deprecated
SEP = 'xxsep'

SPECIAL_TOKS = [BOS, PAD, EOS, S2SCLS, MASK, CSEQ, MSEQ, NSCLS, SEP]

NOTE_TOKS = [f'n{i}' for i in range(pre_process.NOTE_SIZE)] 
DUR_TOKS = [f'd{i}' for i in range(pre_process.DUR_SIZE)]
NOTE_START, NOTE_END = NOTE_TOKS[0], NOTE_TOKS[-1]
DUR_START, DUR_END = DUR_TOKS[0], DUR_TOKS[-1]

MTEMPO_SIZE = 10
MTEMPO_OFF = 'mt0'
MTEMPO_TOKS = [f'mt{i}' for i in range(MTEMPO_SIZE)]

SEQType = Enum('SEQType', 'Mask, Sentence, Melody, Chords, Empty')

learning_rate = 5e-3

checkpoint_dir = 'train_model'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")    
optimizer = tf.keras.optimizers.Adam(learning_rate)

#some extra functions mehhhhhhhhhhhhhhhhhhhh#######################################################    
    
def get_all_midi_dir(root_dir):
    all_midi = []
    for dirName, _, fileList in os.walk(root_dir):
        for fname in fileList:
            if '.mid' in fname:
                all_midi.append(dirName + '/' + fname)

    return all_midi

def make_vocab():
    itos = SPECIAL_TOKS + NOTE_TOKS + DUR_TOKS + MTEMPO_TOKS
    vocab = dict((token, number) for number, token in enumerate(itos))
    return vocab

def make_vocab_reverse():
    itos = SPECIAL_TOKS + NOTE_TOKS + DUR_TOKS + MTEMPO_TOKS
    vocab_reverse = dict((number,token) for number,token in enumerate(itos))
    return vocab_reverse

####################### Model work starts ###############################

def get_batch(vectorized_songs, seq_length, batch_size):
  # the length of the vectorized songs string
  n = vectorized_songs.shape[0] - 1
  # randomly choose the starting indices for the examples in the training batch
  idx = np.random.choice(n-seq_length, batch_size)

  input_batch = [vectorized_songs[i : i+seq_length] for i in idx]


  output_batch = [vectorized_songs[i+1 : i+seq_length+1] for i in idx]
  # output_batch = # TODO
  # x_batch, y_batch provide the true inputs and targets for network training
  x_batch = np.reshape(input_batch, [batch_size, seq_length])
  y_batch = np.reshape(output_batch, [batch_size, seq_length])

  return x_batch, y_batch

def compute_loss(labels, logits):
  loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
  return loss
    
def train_step(x, y , model): 
  # Use tf.GradientTape()
  with tf.GradientTape() as tape:
    y_hat = model(x)
    loss = compute_loss(y, y_hat) 
  
  grads = tape.gradient(loss, model.trainable_variables) 
  # Apply the gradients to the optimizer so it can update the model accordingly
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  return loss

def generate_text(model, random_notes, generation_length ,vocab,vocab_r):
  input_eval = [vocab[n] for n in random_notes] 
  input_eval = tf.expand_dims(input_eval, 0)

  text_generated = []
  model.reset_states()
  tqdm._instances.clear()

  for i in tqdm(range(generation_length)):
      '''TODO: evaluate the inputs and generate the next character predictions'''
      predictions = model(input_eval)
      predictions = tf.squeeze(predictions, 0)
      
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      
      input_eval = tf.expand_dims([predicted_id], 0)
      text_generated.append(vocab_r[predicted_id]) # TODO 
  
  return random_notes + text_generated

def LSTM(rnn_units): 
  return tf.keras.layers.LSTM(
    rnn_units, 
    return_sequences=True, 
    recurrent_initializer='glorot_uniform',
    recurrent_activation='sigmoid',
    stateful=True,
  )

def create_model(vocab_size, embedding_dim, rnn_units, batch_size):
    
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
    LSTM(rnn_units),
    tf.keras.layers.Dense(vocab_size)
  ])

  return model


def main():
    #required lists
    chordarr_list = []
    npenc_list = []
    
    #read multiple files 
    for file_name in get_all_midi_dir(path):
        print('Now loading : \n',file_name)
        mf = midifile.file2mf(file_name)
        stream = midifile.mf2stream(mf)
        chordarr = pre_process.stream2chordarr(stream)
        chordarr_list.append(chordarr)
        
    print("All files Loaded Now Preprocessing.............\n")
    
    #some more preprocessing of every chardarr
    for ca in chordarr_list:
        npenc = pre_process.chordarr2npenc(ca)
        npenc_list.append(npenc)
        
    temp = chordarr_list[0]
    print(temp.shape)
        
    #making the required dictionary 
    vocab = make_vocab()
    vocab_r = make_vocab_reverse()
    #vocab size 
    vocab_size = len(vocab)
    
    #now we will use this dictionary to convert the npenc compressed array
    final_list = []
    for npenc in npenc_list:
        final_list.append(BOS)
        final_list.append(PAD)
        
        for i in range(len(npenc)):
            if(npenc[i][0] == -1):
                 x = SEP
            else:
                x = 'n' + str(npenc[i][0])
            y = 'd' + str(npenc[i][1])
            final_list.append(x)
            final_list.append(y)
        
        final_list.append(PAD)
        final_list.append(EOS)
        
        
    vectorized_list = np.array([vocab[i] for i in final_list])
    #the final list or sequence 

     
    #training our model 
    model = create_model(vocab_size, embedding_dim=256, rnn_units=1024, batch_size=32)
    model.summary()
    
    x, y = get_batch(vectorized_list, seq_length=100, batch_size=32)
    print(x.shape)
    pred = model(x)
    print("Input shape:      ", x.shape, " # (batch_size, sequence_length)")
    print("Prediction shape: ", pred.shape, "# (batch_size, sequence_length, vocab_size)")
    
    sampled_indices = tf.random.categorical(pred[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
    
    example_batch_loss = compute_loss(y, pred)
    
    print("Prediction shape: ", pred.shape, " # (batch_size, sequence_length, vocab_size)") 
    print("scalar_loss:      ", example_batch_loss.numpy().mean())
    
    # Model parameters: 
    embedding_dim = 256 
    rnn_units = 1024  # Experiment between 1 and 2048
    num_training_iterations = 2000 # Increase this to train longer
    batch_size = 64  # Experiment between 1 and 64
    seq_length = 128  # Experiment between 50 and 500
#
    #training our model manually 
    ## Define optimizer and training operation ###
    model = create_model(vocab_size, embedding_dim, rnn_units, batch_size)
    ##################
    # Begin training!#
    ##################
    
    history = []
    plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
    if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists
    
    for iter in tqdm(range(num_training_iterations)):
    
      # Grab a batch and propagate it through the network
      x_batch, y_batch = get_batch(vectorized_list, seq_length, batch_size)
      loss = train_step(x_batch, y_batch, model)
    
      # Update the progress bar
      history.append(loss.numpy().mean())
      plotter.plot(history)
    
      # Update the model with the changed weights!
      if iter % 100 == 0:     
        model.save_weights(checkpoint_prefix)
        
    # Save the trained model and the weights
    model.save_weights(checkpoint_prefix)
    
    model = create_model(vocab_size, embedding_dim, rnn_units, batch_size=1) # TODO    
    # Restore the model weights for the last checkpoint after training
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))

    model.summary()
    
############### Saving The model Optional ###########################
#    model.save('my_h5_model_Att.h5')
#    model = models.load_model('my_h5_model_Att.h5')
#################### Prediction #####################################
    
    random_music = final_list[2:100]
    predictions = generate_text(model, random_music, 1000 ,vocab,vocab_r)
    npenc_out = []
    final_output = predictions
    
    if(final_output[0][0] == 'd'):
        pred = final_output[1:]
    else:
        pred = final_output
        
    
    for (i,j) in zip(pred[::2], pred[1::2]):
        temp = []
        if(i == 'xxsep'):
            temp.append(-1)
        else:
            temp.append(int(i[1:]))
        if(j == 'xxsep'):
            temp.append(-1)
        else:
            temp.append(int(j[1:]))
        npenc_out.append(temp)
        
        
    s = pre_process.npenc2stream(npenc_out,120)
    s.write('midi', fp='output/Gen1.mid')
        
 
if __name__ == "__main__":
    main()

