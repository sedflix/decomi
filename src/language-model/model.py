import numpy as np
from keras.layers import Embedding, LSTM, RNN, Bidirectional, Concatenate, Input, TimeDistributed, Dense, Lambda
from keras.models import Model
from layers import GirNetTwoCell
from utils import sample_logits, sparse_loss

"""
All of the Language Models are *not* supposed to interact with raw words
"""


class LM(object):
    def __init__(self, vocab_size, embedding_size, lstm_hidden_units, vocab):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.lstm_hidden_units = lstm_hidden_units
        self.vocab = vocab
        self.model = None
        self.embedding_layer = None
        self.sent_embedding_model = None

    def build_model(self):
        pass

    def compile_model(self):
        pass

    def load(self, path):
        pass

    def save(self, path):
        pass

    def predict(self, x):
        pass

    def perplexity(self, x):
        pass

    def train(self, x, y):
        pass

    def _build_sent_embedding_model(self):
        pass

    def get_sentence_embedding(self, sentences):
        pass

    def prepare(self, params, samples):
        """
        For evaluation purpose: https://github.com/facebookresearch/SentEval

        * *params*: senteval parameters.
        * *samples*: list of all sentences from the tranfer task.
        * *output*: No output. Arguments stored in "params" can further be used by *batcher*.
        :param params:
        :param samples:
        :return:
        """
        pass

    def batcher(self, params, batch):
        """
        For evaluation purpose: https://github.com/facebookresearch/SentEval

        * *params*: senteval parameters.
        * *batch*: numpy array of text sentences (of size params.batch_size)
        * *output*: numpy array of sentence embeddings (of size params.batch_size)
        :param params:
        :param batch:
        :return:
        """
        pass

    def get_word_embeddings(self, word):
        pass

    def generate(self, seed, size, temperature, out_num=3):
        pass

    def p(self):
        pass


class GirNetTwoLanguageModel(LM):

    def __init__(self, vocab_size, embedding_size, lstm_hidden_units, vocab=None):
        super().__init__(vocab_size, embedding_size, lstm_hidden_units, vocab)

        self.build_model()
        self.compile_model()

    def build_model(self):
        """
        Code used from: https://github.com/divamgupta/mtl_girnet
        :return:
        """
        embed = Embedding(self.vocab_size,
                          self.embedding_size,
                          mask_zero=True,
                          name='embedding_layer')

        rnn_lang1 = LSTM(self.lstm_hidden_units, return_sequences=True, name='rnn_lang1')
        rnn_lang2 = LSTM(self.lstm_hidden_units, return_sequences=True, name='rnn_lang2')

        # lang
        input_lang1 = Input((None,))
        x = embed(input_lang1)
        x = rnn_lang1(x)
        output_lang1 = TimeDistributed(Dense(self.vocab_size, activation='linear', name='lang1_out'))(x)

        # hi
        input_lang2 = Input((None,))
        x = embed(input_lang2)
        x = rnn_lang2(x)
        output_lang2 = TimeDistributed(Dense(self.vocab_size, activation='linear', name='lang2_out'))(x)

        cell_combined = GirNetTwoCell(rnn_lang1.cell, rnn_lang2.cell, self.lstm_hidden_units, name='girnet_cell')

        input_combined = Input((None,))
        x = embed(input_combined)

        x_att = x
        x_att = Bidirectional(LSTM(32, return_sequences=True))(x)
        bider_h = x_att
        x_att = TimeDistributed(Dense(3, activation='softmax'))(x_att)
        x_att = Lambda(lambda x: x[..., 1:])(x_att)

        x = Concatenate(-1)([x_att, x])

        x = RNN(cell_combined, return_sequences=True, name='rnn_prim')(x)
        output_combined = TimeDistributed(Dense(self.lstm_hidden_units, activation='linear', name='prim_out'))(x)

        model = Model([input_lang1, input_lang2, input_combined], [output_lang1, output_lang2, output_combined])

        self.model = model
        self.embedding_layer = embed

    def compile_model(self):
        self.model.compile(loss=sparse_loss, optimizer='adam')

    def p(self):
        self.model.summary()

    def save(self, file_path):
        # TODO: Save the whole model. Naming and vocab and stuffs
        self.model.save_weights(file_path)

    def load(self, file_path):
        self.model.load_weights(file_path)

    def _build_sent_embedding_model(self):
        self.sent_embedding_model = Model(inputs=self.model.input, outputs=self.model.get_layer('rnn_prim').output)

    def get_sentence_embedding(self, sentences, batch_size=64):
        if self.sent_embedding_model is None:
            self._build_sent_embedding_model()
        # TODO: whatever preprocessing is required
        return self.sent_embedding_model.predict([sentences, sentences, sentences], batch_size=batch_size)

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


if __name__ == '__main__':
    lm = GirNetTwoLanguageModel(100, 100, 100)
    lm.p()
