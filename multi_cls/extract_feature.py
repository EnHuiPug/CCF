import os
import json
import random
import argparse
import numpy as np
import pandas as pd
from os.path import basename
from datetime import datetime
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Dense,Lambda
from keras.models import Model
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 


def main():
    parser = argparse.ArgumentParser(description="Run feature extractor")
    parser.add_argument('--maxlen', default=512, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--source_data', default='./data/labeled_data.csv')
    parser.add_argument('--pretrain_model', default='/home/david/pretrain_model/google_bert/chinese_L-12_H-768_A-12')
    parser.add_argument('--finetune_model', default='./best_model.weights')
    parser.add_argument('--finetune', default=False, type=bool)
    parser.add_argument('--layer_name', default='Transformer-11-FeedForward-Norm')
    parser.add_argument('--task', default='labeled')
    args = parser.parse_args()
    print(args)

    maxlen = args.maxlen
    source_data_path = args.source_data

    pretrain_model = args.pretrain_model
    config_path = os.path.join(pretrain_model, 'bert_config.json')
    checkpoint_path = os.path.join(pretrain_model, 'bert_model.ckpt')
    dict_path = os.path.join(pretrain_model, 'vocab.txt')

    label2id = {'时政': 0, '房产': 1, '财经': 2, '科技': 3, '时尚': 4, '教育': 5, '家居': 6}

    def build_model():
        learning_rate = 1e-5
        bert = build_transformer_model(
            config_path,
            checkpoint_path,
            return_keras_model=False,
        )

        output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
        output = Dense(
            units=len(label2id),
            activation='softmax',
            kernel_initializer=bert.initializer
        )(output)

        model = keras.models.Model(bert.model.input, output)
        # model.summary()

        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate),  # 用足够小的学习率
            metrics=['accuracy'],
        )
        return model

    def load_content(filename):
        df = pd.read_csv(filename)
        text = []
        label = []
        if args.task == 'labeled':
            for row in df.itertuples():
                text.append(row.content)
                label.append(label2id[row.class_label])
        else:
            for t in df['content']:
                text.append(t)

        return text, label

    data, label = load_content(source_data_path)

    # 建立分词器
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    if args.finetune:
        model = build_model()
        model.load_weights(args.finetune_model)
        model = Model(inputs=model.input, outputs=model.get_layer(args.layer_name).output)

        model.summary()
    else:    
        model = build_transformer_model(
                config_path,
                checkpoint_path
        )
        model.summary()

    cls_vectors = []
    mean_vectors = []
    for text in tqdm(data):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        cls_fea = model.predict([np.array([token_ids]), np.array([segment_ids])])[0][0]
        mean_fea = np.mean(model.predict([np.array([token_ids]), np.array([segment_ids])])[0], axis=0)
        assert cls_fea.shape[0] == mean_fea.shape[0]
        cls_vectors.append(cls_fea)
        mean_vectors.append(mean_fea)

    print('Save data')
    np.savetxt('./output/{}_cls_features.txt'.format('pretrain' if not args.finetune else 'finetune'), cls_vectors)
    np.savetxt('./output/{}_mean_features.txt'.format('pretrain' if not args.finetune else 'finetune'), mean_vectors)
    if args.task == 'labeled':
        np.savetxt('./output/labels.txt', np.array(label))        

if __name__=='__main__':
    main()