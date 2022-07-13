import streamlit as st 
from multiprocessing import allow_connection_pickling
import numpy as np 
import deep_music_genre
import deep_music_s2s
import deep_music_remix
from fastai.text.models.transformer import tfmerXL_lm_config, Activation
# from .vocab import MusicVocab

# https://github.com/fastai/fastai1/blob/master/fastai/text/models/transformer.py#L175


def default_config():
    config = tfmerXL_lm_config.copy()
    config['act'] = Activation.GeLU

    config['mem_len'] = 512
    config['d_model'] = 512
    config['d_inner'] = 2048
    config['n_layers'] = 6
    config['n_heads'] = 8
    config['d_head'] = 64

    return config

def music_config():
    config = tfmerXL_lm_config.copy()
    config['act'] = Activation.GeLU

    config['mem_len'] = 512
    config['ctx_len'] = 512
    config['d_model'] = 512
    config['d_inner'] = 2048
    config['n_layers'] = 6
    config['n_heads'] = 8
    config['d_head'] = 64

    return config

def btp_phase1_config():
    config = default_config()
    config['act'] = Activation.GeLU

    config['ctx_len'] = 512
    config['d_model'] = 512
    config['d_inner'] = 3072
    config['n_heads'] = 12
    config['d_head'] = 64
    config['n_layers'] = 8
    config['transpose_range'] = (0, 12)
    config['mask_steps'] = 4
    config['encode_position'] = False
    return config

def multitask_config():
    config = music_config()

    config['encode_position'] = True
    config['bias'] = True
    config['enc_layers'] = 10
    config['dec_layers'] = 10
    del config['n_layers']
    return config


#Inputs:
#ckpt_path : Path to Lakh Genre Model checkpoint
@st.cache(hash_funcs={deep_music_genre.MusicLearner: lambda _: None})
def createGenreContinuationModel(encode_position = False, ckpt_path = './checkpoints/lakh_genre_model.pth'):
    config = btp_phase1_config()
    config['transpose_range'] = (0, 12)
    config['mask_steps'] = 4
    config['encode_position'] = False
    return deep_music_genre.music_model_learner(deep_music_genre.MusicDataBunch.empty(''), config=config.copy(), \
                                encode_position = encode_position, pretrained_path = ckpt_path )

@st.cache(hash_funcs={deep_music_remix.MultitaskLearner: lambda _: None})
def createRemixModel(encode_position = True, ckpt_path = './checkpoints/mask_music_model.pth'):
    config = multitask_config()
    return deep_music_remix.multitask_model_learner(deep_music_genre.MusicDataBunch.empty(''), config=config.copy(), \
                                            pretrained_path = ckpt_path )

#Inputs:
# genre_model_learner : returned by createGenreContinuationModel
# mid_file : Expected to be file path!!!

#Notes: 
# mid_file : Expected to be file path!!!
# genre, output_bpm and temperature_ins sliders/dropdown are to be added to the genre interface 
def predictNwGenreModel(genre_model_learner, mid_file, genre = ' POP ', temperature_notes = 1.8, \
    temperature_duration = 1.8, temperature_ins = 1.0, top_p = 0.3, \
    max_len = 512, cutoff_beat = 32, mem_len = 512, allowed_ins = [], \
    output_bpm = 120):

    genre = genre.lower().strip()
    prefix = None
    if 'pop' in genre:
        prefix = 'xxpop'
    elif 'folk' in genre:
        prefix = 'xxfolk'
    elif 'jazz' in genre:
        prefix = 'xxjazz'
    elif 'rock' in genre:
        prefix = 'xxrock'
    elif 'funk' in genre:
        prefix = 'xxfunk'
    elif 'elec' in genre:
        prefix = 'xxelec'

    # assert prefix is not None

    data_vocab = deep_music_genre.MusicVocab.create()
    genre_model_learner.model.mem_len = mem_len

    item = deep_music_genre.MusicItem.from_file(mid_file, data_vocab)
    seed_item = item.trim_to_beat(cutoff_beat)

    #If the genre is given, set the prefix to corresponding token
    if prefix is not None:
        seed_item.data[0] = data_vocab.stoi[prefix]
    #else remove the prefix token and let the model predict without it
    else:
        seed_item.data = seed_item.data[1:]

    if seed_item.to_text().split(' ')[-1] == 'xxeos':
        seed_item.data = seed_item.data[:(len(seed_item.data) - 1)]

    if allowed_ins == []:
        allowed_ins = None
    else:
        for idx, ins in enumerate(allowed_ins):
            if ins == 'Flute':
                allowed_ins[idx] = 'WoodwindInstrument'
            elif ins == 'Brass':
                allowed_ins[idx] = 'BrassInstrument'
            elif ins == 'Violin':
                allowed_ins[idx] = 'StringInstrument'

    pred, full = genre_model_learner.predict(seed_item, n_words=max_len, temperatures=(temperature_notes,temperature_duration,temperature_ins),\
         min_bars=12, top_k=30, top_p=0.65, allowed_ins = allowed_ins)

    # full.to_stream(bpm = output_bpm).write('midi', fp= './outputs/genre_output.mid')

    return full


def createS2SModel(encode_position = False, ckpt_path = 's2s_PianoBasscorrected_TransRangeNone_PartialSyncCheck_10epochs_lakh_model_ATTEMPT1.pth'):
    config = multitask_config()
    return deep_music_s2s.multitask_model_learner(deep_music_s2s.MusicDataBunch.empty(''), config=config.copy(),\
        pretrained_path = '/content/drive/MyDrive/datasets/BTP/models/s2s_PianoBasscorrected_TransRangeNone_PartialSyncCheck_10epochs_lakh_model_ATTEMPT1.pth')

#Inputs:
# genre_model_learner : returned by createGenreContinuationModel
# mid_file : Expected to be file path!!!

#Notes: 
# mid_file : Expected to be file path!!!
# genre, output_bpm and temperature_ins sliders/dropdown are to be added to the genre interface 
def predictMaskModel(mask_model_learner, mid_file, genre = ' POP ', temperature_notes = 1.0, \
    temperature_duration = 1.0, top_p = 0.3, \
    cutoff_beat = 32, output_bpm = 120, \
    pred_type = 'notes', mask_proportion = 0.6):

    genre = genre.lower().strip()
    prefix = None
    if 'pop' in genre:
        prefix = 'xxpop'
    elif 'folk' in genre:
        prefix = 'xxfolk'
    elif 'jazz' in genre:
        prefix = 'xxjazz'
    elif 'rock' in genre:
        prefix = 'xxrock'
    elif 'funk' in genre:
        prefix = 'xxfunk'
    elif 'elec' in genre:
        prefix = 'xxelec'

    # assert prefix is not None

    data_vocab = deep_music_genre.MusicVocab.create()

    item = deep_music_genre.MusicItem.from_file(mid_file, data_vocab)
    seed_item = item.trim_to_beat(cutoff_beat)

    #If the genre is given, set the prefix to corresponding token
    if prefix is not None:
        seed_item.data[0] = data_vocab.stoi[prefix]
    #else remove the prefix token and let the model predict without it
    else:
        seed_item.data = seed_item.data[1:]

    # Remove the eos token if encountered 
    if seed_item.to_text().split(' ')[-1] == 'xxeos':
        seed_item.data = seed_item.data[:(len(seed_item.data) - 1)]

    # Mask the notes / duration before prediction of the seed item 
    if(pred_type == 'notes'):
        # mask 60% of note tokens in the seed item 
        note_indices = [i for i,x in enumerate(data_vocab.textify(seed_item.data).split(' ')) if x[0] == 'n']
        selected_indices = np.random.choice(note_indices, int(len(note_indices) * mask_proportion), replace = False)
        seed_item.data[selected_indices] = data_vocab.mask_idx

        pred = mask_model_learner.predict_mask(seed_item, temperatures=(temperature_notes,temperature_duration))
    else:
        # mask 60% of duration tokens in the seed item
        duration_indices = [i for i,x in enumerate(data_vocab.textify(seed_item.data).split(' ')) if x[0] == 'd']
        selected_indices = np.random.choice(duration_indices, int(len(duration_indices) * mask_proportion), replace = False)
        seed_item.data[selected_indices] = data_vocab.mask_idx

        pred = mask_model_learner.predict_mask(seed_item, temperatures=(0.8,0.8), top_k=40, top_p=0.6)
    
    # full.to_stream(bpm = output_bpm).write('midi', fp= './outputs/genre_output.mid')

    return pred

#TODO: Maybe S2SModel generation between Piano and Bass? But results would most likely not be good so not suitable for presentation/demo