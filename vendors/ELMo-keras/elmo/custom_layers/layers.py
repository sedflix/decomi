from keras.layers import Layer
from keras import backend as K


class GirNetTwoCell(Layer):
    """
    Code used from: https://github.com/divamgupta/mtl_girnet
    """

    def __init__(self, cell_1, cell_2, hidden_units, **kwargs):
        """
        Refer to: https://github.com/divamgupta/mtl_girnet
        :param cell_1: lang1 rnn cell
        :param cell_2: lang2 rnn cell
        :param hidden_units: 
        :param kwargs:
        """
        self.cell_1 = cell_1
        self.cell_2 = cell_2
        self.hidden_units = hidden_units
        self.state_size = [hidden_units, hidden_units]
        super(GirNetTwoCell, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape_n = (input_shape[0], input_shape[1] - 2)

        self._trainable_weights += self.cell_1.trainable_weights
        self._trainable_weights += self.cell_2.trainable_weights

        self._non_trainable_weights += self.cell_1.non_trainable_weights
        self._non_trainable_weights += self.cell_2.non_trainable_weights

        self.built = True

    def call(self, inputs, states):
        hidden_units = self.hidden_units

        gate_val_1 = inputs[:, 0:1]
        gate_val_2 = inputs[:, 1:2]

        inputs = inputs[:, 2:]

        gate_val_1 = K.repeat_elements(gate_val_1, hidden_units, -1)  # shape # bs , hidden
        gate_val_2 = K.repeat_elements(gate_val_2, hidden_units, -1)  # shape # bs , hidden

        _, [h1, c1] = self.cell_1.call(inputs, states)
        _, [h2, c2] = self.cell_2.call(inputs, states)

        h = gate_val_1 * h1 + gate_val_2 * h2 + (1 - gate_val_1 - gate_val_2) * states[0]
        c = gate_val_1 * c1 + gate_val_2 * c2 + (1 - gate_val_1 - gate_val_2) * states[1]

        return h, [h, c]
