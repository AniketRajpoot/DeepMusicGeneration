from encodings import *
import time
import os
import shutil
from vocab import * 
from primitives import *

discarded_path = "/content/drive/MyDrive/datasets/discarded"
data_vocab = MusicVocab.create()

#DONE
def fastai_num_track_filter (arg, num_ins_thresh = 1):
  global time_taken_avg, processed_files

  t1 = time.time()
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
    os.makedirs(discarded_path, exist_ok = True) 
    shutil.move(arg, os.path.join(discarded_path, os.path.basename(arg)))

    return False 
  
  # Check for no. of instruments 
  # print(item.ins)
  if (item.ins is not None) and (len(item.ins.keys()) >= num_ins_thresh):
    print('\t file accepted : ', arg)
    print(f'\t {item.ins}')
    return True
  # Else if we did not store the track -> instrument dictionary from MIDI file
  elif item.ins is None:
    lst =  list(item.data)
    cond_lst = [True if ( (x >= data_vocab.ins_range[0] and x < data_vocab.ins_range[1]) or (x == data_vocab.stoi['xxni']) ) else False for x in lst]
    ins_idxs = item.data[cond_lst]
    uniq_ins = np.unique(ins_idxs)
    num_ins = len(uniq_ins)
    if num_ins >= num_ins_thresh:
      print('\t file accepted : ', arg)
      print(f'\t {[data_vocab.itos[x] for x in uniq_ins]}')
      return True
    else:
      print('\t file discarded due to less instruments : ', arg)
      return False

  else:
    print('\t file discarded due to less instruments : ', arg)
    return False
