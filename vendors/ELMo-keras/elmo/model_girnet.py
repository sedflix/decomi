import json
import os

import numpy as np
import time
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.constraints import MinMaxNorm
from keras.layers import Dense, Input, SpatialDropout1D, Bidirectional, Concatenate, RNN
from keras.layers import LSTM, Activation
from keras.layers import Lambda, Embedding, Conv2D, GlobalMaxPool1D
from keras.layers import add, concatenate
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, load_model
from keras.optimizers import Adagrad
from keras.utils import to_categorical

from data import MODELS_DIR
from .custom_layers import TimestepDropout, Camouflage, Highway, SampledSoftmax
from .custom_layers.layers import GirNetTwoCell


def sample_logits(preds, temperature=1.0):
    """

    Sample an index from a logit vector.

    :param preds:
    :param temperature:
    :return:
    """
    from scipy.misc import logsumexp
    preds = np.asarray(preds).astype('float64')

    if temperature == 0.0:
        return np.argmax(preds)

    preds = preds / temperature
    preds = preds - logsumexp(preds)

    choice = np.random.choice(len(preds), 1, p=np.exp(preds))

    return choice


class ELMo(object):
    def __init__(self, parameters):
        self._model = None
        self._elmo_model = None
        self.parameters = parameters
        self.compile_elmo()

    def __del__(self):
        K.clear_session()
        del self._model

    def char_level_token_encoder(self):
        charset_size = self.parameters['charset_size']
        char_embedding_size = self.parameters['char_embedding_size']
        token_embedding_size = self.parameters['hidden_units_size']
        n_highway_layers = self.parameters['n_highway_layers']
        filters = self.parameters['cnn_filters']
        token_maxlen = self.parameters['token_maxlen']

        # Input Layer, word characters (samples, words, character_indices)
        inputs = Input(shape=(None, token_maxlen,), dtype='int32')
        # Embed characters (samples, words, characters, character embedding)
        embeds = Embedding(input_dim=charset_size, output_dim=char_embedding_size)(inputs)
        token_embeds = []
        # Apply multi-filter 2D convolutions + 1D MaxPooling + tanh
        for (window_size, filters_size) in filters:
            convs = Conv2D(filters=filters_size, kernel_size=[window_size, char_embedding_size], strides=(1, 1),
                           padding="same")(embeds)
            convs = TimeDistributed(GlobalMaxPool1D())(convs)
            convs = Activation('tanh')(convs)
            convs = Camouflage(mask_value=0)(inputs=[convs, inputs])
            token_embeds.append(convs)
        token_embeds = concatenate(token_embeds)
        # Apply highways networks
        for i in range(n_highway_layers):
            token_embeds = TimeDistributed(Highway())(token_embeds)
            token_embeds = Camouflage(mask_value=0)(inputs=[token_embeds, inputs])
        # Project to token embedding dimensionality
        token_embeds = TimeDistributed(Dense(units=token_embedding_size, activation='linear'))(token_embeds)
        token_embeds = Camouflage(mask_value=0)(inputs=[token_embeds, inputs])

        token_encoder = Model(inputs=inputs, outputs=token_embeds, name='token_encoding')
        return token_encoder

    def compile_elmo(self, print_summary=False):
        """
        Compiles a Language Model RNN based on the given parameters
        """

        word_inputs_lang1 = Input(shape=(None,), name='lang1', dtype='int32')
        next_ids_lang1 = Input(shape=(None, 1), name='next_ids_1',
                               dtype='float32')  # Pass outputs as inputs to apply sampled softmax

        word_inputs_lang2 = Input(shape=(None,), name='lang2', dtype='int32')
        next_ids_lang2 = Input(shape=(None, 1), name='next_ids_2',
                               dtype='float32')  # Pass outputs as inputs to apply sampled softmax

        word_inputs_cm = Input(shape=(None,), name='cm', dtype='int32')
        next_ids_cm = Input(shape=(None, 1), name='next_ids_cm',
                            dtype='float32')  # Pass outputs as inputs to apply sampled softmax

        # Train word embeddings from scratch
        embeddings = Embedding(self.parameters['vocab_size'], self.parameters['hidden_units_size'], trainable=True,
                               name='token_encoding')

        def each_inputs(word_inputs):
            inputs = embeddings(word_inputs)
            drop_inputs = SpatialDropout1D(self.parameters['dropout_rate'])(inputs)
            lstm_inputs = TimestepDropout(self.parameters['word_dropout_rate'])(drop_inputs)
            return lstm_inputs, drop_inputs

        # Apply Embeddings to each input
        lstm_inputs_lang1, drop_inputs_lang1 = each_inputs(word_inputs_lang1)
        lstm_inputs_lang2, drop_inputs_lang2 = each_inputs(word_inputs_lang2)
        lstm_inputs_cm, drop_inputs_cm = each_inputs(word_inputs_cm)

        def lang_(lstm_inputs, drop_inputs, next_ids):
            # LSTM for each language
            lstm = LSTM(units=self.parameters['lstm_units_size'],
                        return_sequences=True, activation="tanh",
                        recurrent_activation='sigmoid',
                        kernel_constraint=MinMaxNorm(-1 * self.parameters['cell_clip'],
                                                     self.parameters['cell_clip']),
                        recurrent_constraint=MinMaxNorm(-1 * self.parameters['cell_clip'],
                                                        self.parameters['cell_clip'])
                        )

            lstm_ = lstm(lstm_inputs)

            lstm_ = Camouflage(mask_value=0)(inputs=[lstm_, drop_inputs])
            proj = TimeDistributed(Dense(self.parameters['hidden_units_size'],
                                         activation='linear',
                                         kernel_constraint=MinMaxNorm(-1 * self.parameters['proj_clip'],
                                                                      self.parameters['proj_clip'])
                                         ))(lstm_)

            # Merge Bi-LSTMs feature vectors with the previous ones
            lstm_inputs = add([proj, lstm_inputs])
            # Apply variational drop-out between BI-LSTM layers
            lstm_inputs = SpatialDropout1D(self.parameters['dropout_rate'])(lstm_inputs)

            sampled_softmax = SampledSoftmax(num_classes=self.parameters['vocab_size'],
                                             num_sampled=int(self.parameters['num_sampled']),
                                             tied_to=embeddings if self.parameters['weight_tying']
                                                                   and self.parameters[
                                                                       'token_encoding'] == 'word' else None)

            return sampled_softmax([lstm_inputs, next_ids]), lstm

        outputs_lang1, lstm_lang1 = lang_(lstm_inputs_lang1, drop_inputs_lang1, next_ids_lang1)
        outputs_lang2, lstm_lang2 = lang_(lstm_inputs_lang2, drop_inputs_lang2, next_ids_lang2)

        cell_combined = GirNetTwoCell(lstm_lang1.cell, lstm_lang2.cell, self.parameters['lstm_units_size'],
                                      name='girnet_cell')
        x_att = Bidirectional(LSTM(32, return_sequences=True))(lstm_inputs_cm)
        x_att = TimeDistributed(Dense(3, activation='softmax'))(x_att)
        x_att = Lambda(lambda x: x[..., 1:])(x_att)
        x = Concatenate(-1)([x_att, lstm_inputs_cm])
        x = RNN(cell_combined, return_sequences=True, name='rnn_prim')(x)
        x = Camouflage(mask_value=0)(inputs=[x, drop_inputs_cm])
        proj = TimeDistributed(Dense(self.parameters['hidden_units_size'], activation='linear',
                                     kernel_constraint=MinMaxNorm(-1 * self.parameters['proj_clip'],
                                                                  self.parameters['proj_clip'])
                                     ))(x)

        x = SpatialDropout1D(self.parameters['dropout_rate'])(add([proj, lstm_inputs_cm]))

        sampled_softmax = SampledSoftmax(num_classes=self.parameters['vocab_size'],
                                         num_sampled=int(self.parameters['num_sampled']),
                                         tied_to=embeddings if self.parameters['weight_tying']
                                                               and self.parameters[
                                                                   'token_encoding'] == 'word' else None)

        outputs_cm = sampled_softmax([x, next_ids_cm])

        self._model = Model(
            inputs=[word_inputs_lang1, next_ids_lang1, word_inputs_lang2, next_ids_lang2, word_inputs_cm, next_ids_cm],
            outputs=[outputs_lang1, outputs_lang2, outputs_cm]
        )

        self._model.compile(optimizer=Adagrad(lr=self.parameters['lr'], clipvalue=self.parameters['clip_value']),
                            loss=None)
        if print_summary:
            self._model.summary()

    def train(self, train_data, valid_data, resume=False):
        # Add callbacks (early stopping, model checkpoint)
        weights_file = os.path.join(MODELS_DIR, "elmo_best_weights.hdf5")
        save_best_model = ModelCheckpoint(filepath=weights_file, monitor='val_loss', verbose=1,
                                          save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(patience=self.parameters['patience'], restore_best_weights=True)

        t_start = time.time()

        # Fit Model
        if resume:
            self.load_temp()
        self._model.fit_generator(train_data,
                                  validation_data=valid_data,
                                  epochs=self.parameters['epochs'],
                                  workers=self.parameters['n_threads']
                                  if self.parameters['n_threads'] else os.cpu_count(),
                                  use_multiprocessing=True
                                  if self.parameters['multi_processing'] else False,
                                  callbacks=[save_best_model])

        print('Training took {0} sec'.format(str(time.time() - t_start)))

    def evaluate(self, test_data):
        temp = self.parameters['num_sampled']
        self.parameters['num_sampled'] = self.parameters['vocab_size']

        def unpad(x, y_true, y_pred):
            y_true_unpad = []
            y_pred_unpad = []
            for i, x_i in enumerate(x):
                for j, x_ij in enumerate(x_i):
                    if x_ij == 0:
                        y_true_unpad.append(y_true[i][:j])
                        y_pred_unpad.append(y_pred[i][:j])
                        break
            return np.asarray(y_true_unpad), np.asarray(y_pred_unpad)

        # Generate samples
        x_all = [[] for _ in range(6)]
        for i in range(len(test_data)):
            test_batch = test_data[i][0]
            for j in range(6):
                x_all[j].extend(test_batch[j])

        # Predict outputs
        if self.parameters['unidirectional']:
            y_pred = self._model.predict(x_all)
            y_true = [x_all[1], x_all[3], x_all[5]]
            for j in [2, 1, 0]:
                y_true[j], y_pred[j] = unpad(x_all[j * 2], y_true[j], y_pred[j])
                print('Langauge Model Perplexity %d: %f' % (j, ELMo.perplexity(y_pred[j], y_true[j])))

    def wrap_multi_elmo_encoder(self, print_summary=False, save=False):
        """
        Wrap ELMo meta-model encoder, which returns an array of the 3 intermediate ELMo outputs
        :param print_summary: print a summary of the new architecture
        :param save: persist model
        :return: None
        """

        elmo_embeddings = list()
        elmo_embeddings.append(concatenate(
            [self._model.get_layer('token_encoding').output, self._model.get_layer('token_encoding').output],
            name='elmo_embeddings_level_0'))
        for i in range(self.parameters['n_lstm_layers']):
            if self.parameters['unidirectional']:
                elmo_embeddings.append(self._model.get_layer('f_block_{}'.format(i + 1)).output)
            else:
                elmo_embeddings.append(concatenate([self._model.get_layer('f_block_{}'.format(i + 1)).output,
                                                    Lambda(function=ELMo.reverse)
                                                    (self._model.get_layer('b_block_{}'.format(i + 1)).output)],
                                                   name='elmo_embeddings_level_{}'.format(i + 1)))

        camos = list()
        for i, elmo_embedding in enumerate(elmo_embeddings):
            camos.append(
                Camouflage(mask_value=0.0, name='camo_elmo_embeddings_level_{}'.format(i + 1))([elmo_embedding,
                                                                                                self._model.get_layer(
                                                                                                    'token_encoding').output]))

        self._elmo_model = Model(inputs=[self._model.get_layer('word_indices').input], outputs=camos)

        if print_summary:
            self._elmo_model.summary()

        if save:
            self._elmo_model.save(os.path.join(MODELS_DIR, 'ELMo_Encoder.hd5'))
            print('ELMo Encoder saved successfully')

    def save(self, sampled_softmax=True, temp=False):
        """
        Persist model in disk
        :param sampled_softmax: reload model using the full softmax function
        :return: None
        """

        self.save_temp()
        if not sampled_softmax:
            self.parameters['num_sampled'] = self.parameters['vocab_size']
        self.compile_elmo()
        if temp:
            self._model.load_weights(os.path.join(MODELS_DIR, self.parameters['name'], 'temp_save_weigts.hd5'))
            print("Elmo loaded")
        else:
            try:
                self._model.load_weights(os.path.join(MODELS_DIR, self.parameters['name'], 'elmo_best_weights.hdf5'))
            except:
                self._model.load_weights(os.path.join(MODELS_DIR, self.parameters['name'], 'temp_save_weigts.hd5'))
            self._model.save(os.path.join(MODELS_DIR, self.parameters['name'], 'ELMo_LM_EVAL.hd5'))
            with open(os.path.join(MODELS_DIR, self.parameters['name'], 'parameters.json'), 'w') as file:
                file.write(json.dumps(self.parameters))

            print('Language Model saved successfully')

    def load(self):
        self._model = load_model(os.path.join(MODELS_DIR, self.parameters['name'], 'ELMo_LM_EVAL.hd5'),
                                 custom_objects={'TimestepDropout': TimestepDropout,
                                                 'Camouflage': Camouflage})

    def save_temp(self):
        self._model.save(os.path.join(MODELS_DIR, self.parameters['name'], 'temp_save.hd5'))
        self._model.save_weights(os.path.join(MODELS_DIR, self.parameters['name'], 'temp_save_weigts.hd5'))

    def load_temp(self):
        self._model.load_weights(os.path.join(MODELS_DIR, self.parameters['name'], 'temp_save_weigts.hd5'))

    def load_elmo_encoder(self):
        self._elmo_model = load_model(os.path.join(MODELS_DIR, self.parameters['name'], 'ELMo_Encoder.hd5'),
                                      custom_objects={'TimestepDropout': TimestepDropout,
                                                      'Camouflage': Camouflage})

    def get_outputs(self, test_data, output_type='word', state='last'):
        """
       Wrap ELMo meta-model encoder, which returns an array of the 3 intermediate ELMo outputs
       :param test_data: data generator
       :param output_type: "word" for word vectors or "sentence" for sentence vectors
       :param state: 'last' for 2nd LSTMs outputs or 'mean' for mean-pooling over inputs, 1st LSTMs and 2nd LSTMs
       :return: None
       """
        # Generate samples
        x = []
        for i in range(len(test_data)):
            test_batch = test_data[i][0]
            x.extend(test_batch[0])

        preds = np.asarray(self._elmo_model.predict(np.asarray(x)))
        if state == 'last':
            elmo_vectors = preds[0]
        else:
            elmo_vectors = np.mean(preds, axis=0)

        if output_type == 'words':
            return elmo_vectors
        else:
            return np.mean(elmo_vectors, axis=1)

    def generate(self, seed, size=32, branch=2):
        for i in range(len(seed), size):
            temp_seed_forward = np.array(seed[:-1] + [0]).reshape(len(seed), 1, 1)
            x_seed = np.array(seed).reshape(len(seed), 1)
            x = [x_seed, temp_seed_forward, x_seed, temp_seed_forward, x_seed, temp_seed_forward]
            pred = self._model.predict(x)
            seed.append(np.argmax(pred[branch][i - 1]))
        return seed

    def generate(self, seed, size, temperature, out_num=3):
        """
        Code used from: https://github.com/pbloem/language-models

        :param seed: The first few wordss of the sequence to start generating from. They must be integers.
        :param size: The total size of the sequence to generate
        :param temperature: This controls how much we follow the probabilities provided by the network. For t=1.0 we just
            sample directly according to the probabilities. Lower temperatures make the high-probability words more likely
            (providing more likely, but slightly boring sentences) and higher temperatures make the lower probabilities more
            likely (resulting is weirder sentences). For temperature=0.0, the generation is _greedy_, i.e. the word with the
            highest probability is always chosen.
        :return: A list of integers representing a samples sentence
        """

        ls = seed.shape[0]

        # Due to the way Keras RNNs work, we feed the model a complete sequence each time. At first it's just the seed,
        # zero-padded to the right length. With each iteration we sample and set the next character.

        # tokens = np.concatenate([seed, np.zeros(size - ls)])
        tokens_all = []
        for i in range(out_num):
            tokens_all.append(np.concatenate([seed, np.zeros(size - ls)]))

        for i in range(ls, size):
            tokens_to_predict = []
            for j in range(out_num):
                tokens_to_predict.append(tokens_all[j][None, :])

            all_probs = self.model.predict(tokens_to_predict)

            # Extract the i-th probability vector and sample an index from it
            for j, probs in enumerate(all_probs):
                next_token = sample_logits(probs[0, i - 1, :], temperature=temperature)
                tokens_all[j][i] = next_token

        return [tokens.astype('int') for tokens in tokens_all]

    @staticmethod
    def reverse(inputs, axes=1):
        return K.reverse(inputs, axes=axes)

    @staticmethod
    def perplexity(y_pred, y_true):
        cross_entropies = []
        for y_pred_seq, y_true_seq in zip(y_pred, y_true):
            # Reshape targets to one-hot vectors
            y_true_seq = to_categorical(y_true_seq, y_pred_seq.shape[-1])
            # Compute cross_entropy for sentence words
            cross_entropy = K.categorical_crossentropy(K.tf.convert_to_tensor(y_true_seq, dtype=K.tf.float32),
                                                       K.tf.convert_to_tensor(y_pred_seq, dtype=K.tf.float32))
            cross_entropies.extend(cross_entropy.eval(session=K.get_session()))

        # Compute mean cross_entropy and perplexity
        cross_entropy = np.mean(np.asarray(cross_entropies), axis=-1)

        return pow(2.0, cross_entropy)
