from tap import Tap

class ModelParser(Tap):
    hidden_size: int = 128  # GNN隐层向量维度
    num_classes: int = 2
    vector_size = 13
    maxLen = 500
    layer_num = 2
    dropout = 0.2
    units = 256

model_args = ModelParser().parse_args(known_only=True)

from keras.models import Sequential
from keras.layers import Masking, Dense, Dropout, GRU, Bidirectional
# from keras.layers.core import Masking, Dense, Dropout
# from keras.layers.recurrent import GRU
# from keras.layers.wrappers import Bidirectional

def build_model():
    print('Build model...')
    model = Sequential()
    model.add(Masking(mask_value=1.0, input_shape=(model_args.maxLen, model_args.vector_size)))

    for i in range(1, model_args.layer_num):
        model.add(Bidirectional(
            GRU(units=model_args.units, activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=True)))
        model.add(Dropout(model_args.dropout))

    model.add(Bidirectional(GRU(units=model_args.units, activation='tanh', recurrent_activation='hard_sigmoid')))
    model.add(Dropout(model_args.dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
    model.summary()

    return model