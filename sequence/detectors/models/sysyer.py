from tap import Tap

class ModelParser(Tap):
    hidden_size: int = 128  # GNN隐层向量维度
    num_classes: int = 2
    vector_size = 30
    maxLen = 500
    layer_num = 2
    dropout = 0.2
    units = 256

model_args = ModelParser().parse_args(known_only=True)

from keras.models import Sequential
from keras.layers import Masking, Dense, Dropout, GRU, Bidirectional

def build_model():
    """
    输入参数：embedding初始weight
    """
    print('Build model...')
    model = Sequential()
    # Embedding trainable参数设置为False
    # model.add(Embedding(weight.shape[0], conf.vectorDim, mask_zero=True, weights=[weight], trainable=False))
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