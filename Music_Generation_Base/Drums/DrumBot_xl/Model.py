import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from enum import Enum
import matplotlib.pyplot as plt
from typing import *
import time

def embedding_lookup(lookup_table, x):
    return tf.compat.v1.nn.embedding_lookup(lookup_table, x)


def normal_embedding_lookup(x, n_token, d_embed, d_proj, initializer,
                            proj_initializer, scope='normal_embed', **kwargs):
    emb_scale = d_proj ** 0.5
    with tf.compat.v1.variable_scope(scope):
        lookup_table = tf.compat.v1.get_variable('lookup_table', [n_token, d_embed], initializer=initializer)
        y = embedding_lookup(lookup_table, x)
        if d_proj != d_embed:
            proj_W = tf.compat.v1.get_variable('proj_W', [d_embed, d_proj], initializer=proj_initializer)
            y = tf.einsum('ibe,ed->ibd', y, proj_W)
        else:
            proj_W = None
        ret_params = [lookup_table, proj_W]
    y *= emb_scale
    return y, ret_params


def normal_softmax(hidden, target, n_token, params, scope='normal_softmax', **kwargs):
    def _logit(x, W, b, proj):
        y = x
        if proj is not None:
            y = tf.einsum('ibd,ed->ibe', y, proj)
        return tf.einsum('ibd,nd->ibn', y, W) + b

    params_W, params_projs = params[0], params[1]

    with tf.compat.v1.variable_scope(scope):
        softmax_b = tf.compat.v1.get_variable('bias', [n_token], initializer=tf.zeros_initializer())
        output = _logit(hidden, params_W, softmax_b, params_projs)
        nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=output)
    return nll, output


def positional_embedding(pos_seq, inv_freq, bsz=None):
    sinusoid_inp = tf.einsum('i,j->ij', pos_seq, inv_freq)
    pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
    if bsz is not None:
        return tf.tile(pos_emb[:, None, :], [1, bsz, 1])
    else:
        return pos_emb[:, None, :]


def positionwise_FF(inp, d_model, d_inner, dropout, kernel_initializer,
                    scope='ff', is_training=True):
    output = inp
    with tf.compat.v1.variable_scope(scope):
        output = tf.keras.layers.Dense(d_inner, activation=tf.nn.relu, 
                                       kernel_initializer=kernel_initializer, name='layer_1')(inp)
        output = tf.keras.layers.Dropout(dropout, name='drop_1')(output, training=is_training)
        output = tf.keras.layers.Dense(d_model, activation=tf.nn.relu, 
                                       kernel_initializer=kernel_initializer, name='layer_2')(output)
        output = tf.keras.layers.Dropout(dropout, name='drop_2')(output, training=is_training)
        output = tf.keras.layers.LayerNormalization(axis=-1)(output + inp)
    return output


def _create_mask(qlen, mlen, same_length=False):
    attn_mask = tf.ones([qlen, qlen])
    mask_u = tf.linalg.band_part(attn_mask, 0, -1)
    mask_dia = tf.linalg.band_part(attn_mask, 0, 0)
    attn_mask_pad = tf.zeros([qlen, mlen])
    ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
    if same_length:
        mask_l = tf.matrix_band_part(attn_mask, -1, 0)
        ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
    return ret


def _cache_mem(curr_out, prev_mem, mem_len=None):
    if mem_len is None or prev_mem is None:
        new_mem = curr_out
    elif mem_len == 0:
        return prev_mem
    else:
        new_mem = tf.concat([prev_mem, curr_out], 0)[-mem_len:]
    return tf.stop_gradient(new_mem)


def rel_shift(x):
    x_size = tf.shape(x)
    x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
    x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, x_size)
    return x


def rel_multihead_attn(w, r, r_w_bias, r_r_bias, attn_mask, mems, d_model,
                       n_head, d_head, dropout, dropatt, is_training,
                       kernel_initializer, scope='rel_attn'):
    scale = 1 / (d_head ** 0.5)
    with tf.compat.v1.variable_scope(scope):
        qlen = tf.shape(w)[0]
        rlen = tf.shape(r)[0]
        bsz = tf.shape(w)[1]

        cat = tf.concat([mems, w], 0) if mems is not None and mems.shape.ndims > 1 else w

        w_heads = tf.keras.layers.Dense(3 * n_head * d_head, use_bias=False, 
                                        kernel_initializer=kernel_initializer, name='qkv')(cat)
        r_head_k = tf.keras.layers.Dense(n_head * d_head, use_bias=False,
                                         kernel_initializer=kernel_initializer, name='r')(r)
        
        w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
        w_head_q = w_head_q[-qlen:]

        klen = tf.shape(w_head_k)[0]

        w_head_q = tf.reshape(w_head_q, [qlen, bsz, n_head, d_head])
        w_head_k = tf.reshape(w_head_k, [klen, bsz, n_head, d_head])
        w_head_v = tf.reshape(w_head_v, [klen, bsz, n_head, d_head])

        r_head_k = tf.reshape(r_head_k, [rlen, n_head, d_head])

        rw_head_q = w_head_q + r_w_bias
        rr_head_q = w_head_q + r_r_bias

        AC = tf.einsum('ibnd,jbnd->ijbn', rw_head_q, w_head_k)
        BD = tf.einsum('ibnd,jnd->ijbn', rr_head_q, r_head_k)
        BD = rel_shift(BD)

        attn_score = (AC + BD) * scale
        attn_mask_t = attn_mask[:, :, None, None]
        attn_score = attn_score * (1 - attn_mask_t) - 1e30 * attn_mask_t

        attn_prob = tf.nn.softmax(attn_score, 1)
        attn_prob = tf.keras.layers.Dropout(dropatt)(attn_prob, training=is_training)

        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
        size_t = tf.shape(attn_vec)
        attn_vec = tf.reshape(attn_vec, [size_t[0], size_t[1], n_head * d_head])

        attn_out = tf.keras.layers.Dense(d_model, use_bias=False, 
                                         kernel_initializer=kernel_initializer, name='o')(attn_vec)
        attn_out = tf.keras.layers.Dropout(dropout)(attn_out, training=is_training)
        output = tf.keras.layers.LayerNormalization(axis=-1)(attn_out + w)
        return output



def transformer(dec_inp, target, mems, n_token, n_layer, d_model, d_embed,
                n_head, d_head, d_inner, dropout, dropatt,
                initializer, is_training, proj_initializer=None,
                mem_len=None, cutoffs=[], div_val=1, tie_projs=[],
                same_length=False, clamp_len=-1,
                input_perms=None, target_perms=None, head_target=None,
                untie_r=False, proj_same_dim=True,
                scope='transformer'):
    """
    cutoffs: a list of python int. Cutoffs for adaptive softmax.
    tie_projs: a list of python bools. Whether to tie the projections.
    perms: a list of tensors. Each tensor should of size [len, bsz, bin_size].
        Only used in the adaptive setting.
    """

    new_mems = []
    with tf.compat.v1.variable_scope(scope, reuse= tf.compat.v1.AUTO_REUSE):
        if untie_r:
            r_w_bias = tf.compat.v1.get_variable('r_w_bias', [n_layer, n_head, d_head], initializer=initializer)
            r_r_bias = tf.compat.v1.get_variable('r_r_bias', [n_layer, n_head, d_head], initializer=initializer)
        else:
            r_w_bias = tf.compat.v1.get_variable('r_w_bias', [n_head, d_head], initializer=initializer)
            r_r_bias = tf.compat.v1.get_variable('r_r_bias', [n_head, d_head], initializer=initializer)

        qlen = tf.shape(dec_inp)[0]
        mlen = tf.shape(mems[0])[0] if mems is not None else 0
        klen = qlen + mlen

        if proj_initializer is None:
            proj_initializer = initializer

        embeddings, shared_params = normal_embedding_lookup(
            x=dec_inp,
            n_token=n_token,
            d_embed=d_embed,
            d_proj=d_model,
            initializer=initializer,
            proj_initializer=proj_initializer)
        
        attn_mask = _create_mask(qlen, mlen, same_length)
        
        pos_seq = tf.range(klen - 1, -1, -1.0)
        if clamp_len > 0:
            pos_seq = tf.minimum(pos_seq, clamp_len)
        inv_freq = 1 / (10000 ** (tf.range(0, d_model, 2.0) / d_model))
        pos_emb = positional_embedding(pos_seq, inv_freq)

        output = tf.keras.layers.Dropout(rate=dropout)(embeddings, training=is_training)
        pos_emb = tf.keras.layers.Dropout(rate=dropout)(pos_emb, training=is_training)

        if mems is None:
            mems = [None] * n_layer

        for i in range(n_layer):
            # cache new mems
            new_mems.append(_cache_mem(output, mems[i], mem_len))

            with tf.compat.v1.variable_scope('layer_{}'.format(i)):
                output = rel_multihead_attn(
                    w=output,
                    r=pos_emb,
                    r_w_bias=r_w_bias if not untie_r else r_w_bias[i],
                    r_r_bias=r_r_bias if not untie_r else r_r_bias[i],
                    attn_mask=attn_mask,
                    mems=mems[i],
                    d_model=d_model,
                    n_head=n_head,
                    d_head=d_head,
                    dropout=dropout,
                    dropatt=dropatt,
                    is_training=is_training,
                    kernel_initializer=initializer)

                output = positionwise_FF(
                    inp=output,
                    d_model=d_model,
                    d_inner=d_inner,
                    dropout=dropout,
                    kernel_initializer=initializer,
                    is_training=is_training)

        # apply Dropout
        output = tf.keras.layers.Dropout(dropout)(output, training=is_training)

        loss, logits = normal_softmax(
            hidden=output,
            target=target,
            n_token=n_token,
            params=shared_params)

        return loss, logits, new_mems

class TransformerXL(object):
    ########################################
    # initialize
    ########################################
    def __init__(self, vocab_size, memlen, checkpoint=None, is_training=False, training_seqs=None):
        # load dictionary
        self.event2word = vocab_size
        # model settings
        self.x_len = 256     #input sequence length
        self.mem_len = memlen    #
        self.n_layer = 6
        self.d_embed = 768
        self.d_model = 768
        self.dropout = 0.1    ##
        self.n_head = 12
        self.d_head = self.d_model // self.n_head
        self.d_ff = 3072
        self.n_token = (self.event2word)
        self.learning_rate = 1e-4      ##
        self.group_size = 3
        self.entry_len = self.group_size * self.x_len
        # mode
        self.is_training = is_training
        self.training_seqs = training_seqs
        self.checkpoint = checkpoint
        if self.is_training: # train from scratch or finetune
            self.batch_size = 8        
        else: # inference
            self.batch_size = 1
        # load model
        self.load_model()

    ########################################
    # load model
    ########################################
    
    def load_model(self):
        tf.compat.v1.disable_eager_execution()
        # placeholders ---> train
        self.x = tf.compat.v1.placeholder(tf.int32, shape=[self.batch_size, None])
        self.y = tf.compat.v1.placeholder(tf.int32, shape=[self.batch_size, None])
        self.mems_i = [tf.compat.v1.placeholder(tf.float32, [self.mem_len, self.batch_size, self.d_model]) for _ in range(self.n_layer)]
        # placeholders ---> test
        self.x_t = tf.compat.v1.placeholder(tf.int32, shape=[1, None])
        self.y_t = tf.compat.v1.placeholder(tf.int32, shape=[1, None])
        self.mems_it = [tf.compat.v1.placeholder(tf.float32, [self.mem_len, 1, self.d_model]) for _ in range(self.n_layer)]
        # model
        self.global_step = tf.compat.v1.train.get_or_create_global_step()

        # initialize parameters
        initializer = tf.compat.v1.initializers.random_normal(stddev=0.02, seed=None)
        proj_initializer = tf.compat.v1.initializers.random_normal(stddev=0.01, seed=None)
        
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
            xx = tf.transpose(self.x, [1, 0])
            yy = tf.transpose(self.y, [1, 0])
            loss, self.logits, self.new_mem = transformer(
                dec_inp=xx,
                target=yy,
                mems=self.mems_i,
                n_token=self.n_token,
                n_layer=self.n_layer,
                d_model=self.d_model,
                d_embed=self.d_embed,
                n_head=self.n_head,
                d_head=self.d_head,
                d_inner=self.d_ff,
                dropout=self.dropout,
                dropatt=self.dropout,
                initializer=initializer,
                proj_initializer=proj_initializer,
                is_training=self.is_training,
                mem_len=self.mem_len,
                cutoffs=[],
                div_val=-1,
                tie_projs=[],
                same_length=False,
                clamp_len=-1,
                input_perms=None,
                target_perms=None,
                head_target=None,
                untie_r=False,
                proj_same_dim=True)
        self.avg_loss = tf.reduce_mean(loss)
        # vars
        all_vars = tf.compat.v1.trainable_variables()
        print ('num parameters:', np.sum([np.prod(v.get_shape().as_list()) for v in all_vars]))
        grads = tf.gradients(self.avg_loss, all_vars)
        grads_and_vars = list(zip(grads, all_vars))
        # gradient clipping
        def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_norm(grad, 100.)
        
        grads_and_vars = [(ClipIfNotNone(grad), var) for grad, var in grads_and_vars]
        all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.compat.v1.trainable_variables()])
        # optimizer
        #warmup_steps = 0
        # increase the learning rate linearly
        #if warmup_steps > 0:
        #    warmup_lr = tf.compat.v1.to_float(self.global_step) / tf.compat.v1.to_float(warmup_steps) \
        #          * self.learning_rate
        #else:
        #    warmup_lr = 0.0

        decay_lr = tf.compat.v1.train.cosine_decay(
            self.learning_rate,
            global_step=self.global_step,
            decay_steps=200000,
            alpha=0.004)
        
        #lr_decay_warmup = tf.where(self.global_step < warmup_steps,
        #                    warmup_lr, decay_lr)
        #decay_lr = tf.compat.v1.train.cosine_decay_warmup(     ##
        #     self.learning_rate,
        #     global_step=self.global_step,
        #     decay_steps=200000,
        #     warmup_steps=16000,
        #     alpha=0.004
        #)
        
        #try:
            #self.optimizer = tfa.optimizers.LAMB(learning_rate=decay_lr)
            #print('LAMBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
        #except:
            #self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=decay_lr)
            #print('ADAMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM')
            #pass
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=decay_lr)
        self.train_op = self.optimizer.apply_gradients(grads_and_vars, self.global_step)
        # saver
        self.saver = tf.compat.v1.train.Saver(max_to_keep=100)
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        # load pre-trained checkpoint or note
        if self.checkpoint:
            self.saver.restore(self.sess, self.checkpoint)
        else:
            self.sess.run(tf.compat.v1.global_variables_initializer())
            
    
            
    ########################################
    # train
    ########################################
    def train(self, training_data, output_checkpoint_folder):
        # check output folder
        if not os.path.exists(output_checkpoint_folder):
            os.mkdir(output_checkpoint_folder)
        # shuffle
        index = np.arange(len(training_data))
        np.random.shuffle(index)
        training_data = training_data[index]
        num_batches = len(training_data) // self.batch_size
        st = time.time()
        for e in range(1000):
            total_loss = []
            for i in range(num_batches):
                segments = training_data[self.batch_size*i:self.batch_size*(i+1)]
                batch_m = [np.zeros((self.mem_len, self.batch_size, self.d_model), dtype=np.float32) for _ in range(self.n_layer)]
                for j in range(self.group_size):
                    batch_x = segments[:, j, 0, :]
                    batch_y = segments[:, j, 1, :]
                    # prepare feed dict
                    feed_dict = {self.x: batch_x, self.y: batch_y}
                    for m, m_np in zip(self.mems_i, batch_m):
                        feed_dict[m] = m_np
                    # run
                    _, gs_, loss_, new_mem_ = self.sess.run([self.train_op, self.global_step, self.avg_loss, self.new_mem], feed_dict=feed_dict)
                    batch_m = new_mem_
                    total_loss.append(loss_)
                    # print ('Current lr: {}'.format(self.sess.run(self.optimizer._lr)))
                    print('>>> Epoch: {}, Step: {}, Loss: {:.5f}, Time: {:.2f}'.format(e, gs_, loss_, time.time()-st))
                    print('i : ',i,' j : ',j)
                    if not i % 500:
                        self.saver.save(self.sess, '{}/model-{:03d}-{:.3f}'.format(output_checkpoint_folder, e, np.mean(total_loss)))
                    

            print ('[epoch {} avg loss] {:.5f}'.format(e, np.mean(total_loss)))
            if not e % 6:
                self.saver.save(self.sess, '{}/model-{:03d}-{:.3f}'.format(output_checkpoint_folder, e, np.mean(total_loss)))
            # stop
            if np.mean(total_loss) <= 0.0001:
                break

    ########################################
    # search strategy: temperature (re-shape)
    ########################################
    def temperature(self, logits, temperature):
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        return probs


    ########################################
    # search strategy: nucleus (truncate)
    ########################################
    def nucleus(self, probs, p):
        probs /= sum(probs)
        sorted_probs = np.sort(probs)[::-1]
        sorted_index = np.argsort(probs)[::-1]
        cusum_sorted_probs = np.cumsum(sorted_probs)
        after_threshold = cusum_sorted_probs > p
        if sum(after_threshold) > 0:
            last_index = np.where(after_threshold)[0][-1]
            candi_index = sorted_index[:last_index]
        else:
            candi_index = sorted_index[:3] # just assign a value
        candi_probs = [probs[i] for i in candi_index]
        candi_probs /= sum(candi_probs)
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word

    ########################################
    # evaluate (for batch size = 1)
    ########################################
    def evaluate(self, notes, num_notes, k, strategies, use_structure=False, init_mem = None):

      batch_size = 1
      # initialize mem
      if init_mem is None:
          batch_m = [np.zeros((self.mem_len, batch_size, self.d_model), dtype=np.float32) for _ in range(self.n_layer)]
      else:
          batch_m = init_mem 

      initial_flag = True
      fail = 0
      i = 0

      while i < num_notes:
            if fail>200:
              print('Fail : ',fail)
              #continue

            # prepare input
            if initial_flag:
                temp_x = np.zeros((batch_size, len(notes[0])))
                for b in range(batch_size):
                    for z, t in enumerate(notes[b]):
                        temp_x[b][z] = t
                initial_flag = False
            else:
                temp_x = np.zeros((batch_size, 1))
                for b in range(batch_size):
                    temp_x[b][0] = notes[b][-1]

            # prepare feed dict
            # inside a feed dict
            # placeholder : data
            # put input into feed_dict
            feed_dict = {self.x: temp_x}

            # put memeory into feed_dict
            for m, m_np in zip(self.mems_i, batch_m):
                feed_dict[m] = m_np
            
            # model (prediction)
            _logits, _new_mem = self.sess.run([self.logits, self.new_mem], feed_dict=feed_dict)
            
            #print('shape : ',_logits.shape)
            logits = _logits[-1, 0]

            # temperature or not
            if k == 0:
              ran = float((np.random.randint(14,16))/10)
            else:
              ran = float((np.random.randint(7,10))/10)
            
            probs = self.temperature(logits=logits, temperature=ran)

            # sampling
            # note : the generated tokenized event
            #ran_n = float((np.random.randint(90,98))/100)
            note = self.nucleus(probs=probs, p=0.90)
            

            if note not in tokenizer.index_word:
              continue
            
            

            # add new event to record sequence
            notes = np.append(notes[0], note)
            notes = np.reshape(notes, (1, len(notes)))
            #print('notes : ',notes.shape)
            
            # re-new mem
            batch_m = _new_mem
            fail = 0
            i += 1

      return notes[0]

    ########################################
        # predict (for batch size = 1)
    ########################################
    def predict(self, notes, num_notes, k, strategies, use_structure=False):
      prediction = self.evaluate(notes, num_notes, k, strategies, use_structure)

      predicted_sentence = []
  
      for i in prediction:
          # print('helllllo',int(i))
          i = int(i)
          if i < len(tokenizer.word_index) and i>0:
              predicted_sentence.append(tokenizer.index_word[i])
      return predicted_sentence

def get_data(notes_chords, sequence_length):
    
    # sequence_length = 100
    notes_input = []
    notes_output = []
    shift = 1
    
    for i in range(0, len(notes_chords) - sequence_length, 1):
        temp_input = ''
        temp_output = ''
        for j in range(i,i + sequence_length):
            temp_input += notes_chords[j] + ' '
        notes_input.append(temp_input)
        for j in range(i+shift,i + sequence_length+shift):
            temp_output += notes_chords[j] + ' '
        notes_output.append(temp_output)


    n_patterns = len(notes_input)
    # notes_normalized_input = np.reshape(notes_input, (n_patterns, sequence_length))
    # notes_normalized_input =  notes_normalized_input / float(n_vocab)
    #notes_output = np.array(notes_output)


    return (notes_input, notes_output)


########################################
    # Prepare data
########################################
        
def xl_data(input_, output, group_size):
        training_data = []
    
        pairs = []
        for i in range(0, len(input_)):
            x, y = input_[i], output[i]
            
            pairs.append([x, y])

        pairs = np.array(pairs)
    
        # put pairs into training data by groups
        for i in range(0, len(pairs) - group_size + 1, group_size):
            segment = pairs[i:i+group_size]
            assert len(segment) == group_size
            training_data.append(segment)
            
        training_data = np.array(training_data)
        
        return training_data