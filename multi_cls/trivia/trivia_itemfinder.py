import os
import gc
import json
import random
import argparse
import collections
import numpy as np
import pandas as pd
from os.path import basename
from datetime import datetime
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Dense, Lambda
from keras.models import Model
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from keras.utils import to_categorical
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score



def main():
    parser = argparse.ArgumentParser(description="Run text classifier")
    parser.add_argument('--train', default=False, type=bool)
    parser.add_argument('--predict', default=True, type=bool)
    parser.add_argument('--maxlen', default=256, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--pos_train_data', default='./train/positive_train')
    parser.add_argument('--neg_train_data', default='./train/negative_train')
    parser.add_argument('--test_data', default='./test/test_data')
    parser.add_argument('--pretrain_model',
                        default='../chinese_L-12_H-768_A-12')
    parser.add_argument('--train_data_split_ratio', default=0.25, type=float)
    parser.add_argument('--ensamble', default=False, type=bool)
    parser.add_argument('--timestamp', default='')
    parser.add_argument('--replace_emoji', default=True, type=bool)
    parser.add_argument('--balance_labeled_data', default=False, type=bool)
    parser.add_argument('--patience', default=5)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    args = parser.parse_args()
    print(args)

    # settings
    maxlen = args.maxlen
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    pretrain_model = args.pretrain_model
    config_path = os.path.join(pretrain_model, 'bert_config.json')
    checkpoint_path = os.path.join(pretrain_model, 'bert_model.ckpt')
    dict_path = os.path.join(pretrain_model, 'vocab.txt')

    # data
    pos_train_data_path = args.pos_train_data
    neg_train_data_path = args.neg_train_data
    test_data_path = args.test_data

    def load_data(file):
        data = []
        labels = []
        with open(file) as fr:
            for line in fr:
                answer, question, sent = line.split("\t")
                data.append(sent.strip())
                if os.path.basename(file) == "negative_train":
                    labels.append(0)
                else:
                    labels.append(1)

        return data, labels

    def load_test(file):
        data = []
        with open(file) as f:
            for line in f:
                # answer, question, sent = line.split("\t")
                # data.append((answer, question, sent.strip()))

                # bkid, rank, question, answer, distracts = line.split("\t")
                # data.append((bkid, rank, question, answer, distracts.strip()))

                answer, question, sent, status = line.split("\t")
                data.append((answer, question, sent, status.strip()))
        return data

    def balance_labeled_data(data, labels):
        pos = []
        neg = []
        for d, l in zip(data, labels):
            if l == 1:
                pos.append((d, l))
            else:
                neg.append((d, l))
        if len(pos) >= len(neg):
            pos_ = random.sample(pos, len(neg))
            neg_ = neg
        else:
            pos_ = pos
            neg_ = random.sample(neg, len(pos))
        data = [i[0] for i in pos_] + [i[0] for i in neg_]
        labels = [i[1] for i in pos_] + [i[1] for i in neg_]
        return data, labels

    # 建立分词器
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    class data_generator(DataGenerator):
        """数据生成器
        """

        def __iter__(self, random=False):
            batch_token_ids, batch_segment_ids, batch_labels = [], [], []
            for is_end, (text, label) in self.sample(random):
                token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([label])
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_labels = sequence_padding(batch_labels)
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []

    def build_model():
        bert = build_transformer_model(
            config_path,
            checkpoint_path,
            return_keras_model=False,
        )

        output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
        output = Dense(
            units=2,
            activation='softmax',
            kernel_initializer=bert.initializer
        )(output)

        model = keras.models.Model(bert.model.input, output)
        model.summary()

        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate),  # 用足够小的学习率
            metrics=['accuracy'],
        )

        return model

    def evaluate(data):
        f1, precision, recall, accuracy = 0., 0., 0., 0.,
        y, y_ = [], []

        for x_true, y_true in data:
            y_p = model.predict(x_true).argmax(axis=-1)
            y.extend([y[0] for y in y_true])
            y_.extend(y_p)
        print("y len, y_ len: {}, {}".format(len(y), len(y_)))
        f1 = f1_score(y, y_)
        precision = precision_score(y, y_)
        recall = recall_score(y, y_)
        accuracy = accuracy_score(y, y_)
        return f1, precision, recall, accuracy, y_

    def predict(data):
        y_hat = []
        for x_true, _ in tqdm(data):
            y_p = model.predict(x_true)
            y_hat.extend([x[1] for x in y_p])
            # y_hat.extend(y_p)
        return y_hat

    class Evaluator(keras.callbacks.Callback):
        """评估与保存
        """

        def __init__(self):
            super(keras.callbacks.Callback, self).__init__()
            self.best_val_f1 = 0.
            self.best_epoch = 0
            self.metric_history = {}
            self.wait = 0
            self.pred = None

        def on_epoch_end(self, epoch, logs=None):
            f1, precision, recall, accuracy, y_ = evaluate(valid_generator)
            self.metric_history[epoch] = {'f1': f1, 'precision': precision, 'recall': recall, 'accuracy': accuracy}
            if f1 > self.best_val_f1:
                self.pred = y_
                self.wait = 0
                self.best_val_f1 = f1
                self.best_epoch = epoch
                self.metric_history['best'] = {'best_val_f1': self.best_val_f1, 'best_epoch': self.best_epoch}
                self.model.save_weights('best_model.weights')
            print(
                u'val_f1: %.5f, 'u'val_precision: %.5f, 'u'val_recall: %.5f, 'u'val_accuracy: %.5f, best_val_f1: %.5f\n' %
                (f1, precision, recall, accuracy, self.best_val_f1)
            )
            self.wait += 1
            if self.wait > args.patience:
                self.model.stop_training = True

    assert args.train or args.predict or args.ensamble == True

    if args.train:
        print('Training process')
        expr = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # 读取数据
        p_train_data, p_train_label = load_data(pos_train_data_path)
        n_train_data, n_train_label = load_data(neg_train_data_path)

        train_data = p_train_data + n_train_data
        train_label = p_train_label + n_train_label

        print("training data 0 : {}".format(train_data[0]))
        print("training data length : {}".format(len(train_data)))

        X_train, X_val, y_train, y_val = train_test_split(train_data, train_label,
                                                          test_size=args.train_data_split_ratio, random_state=42)

        trains = [(t, l) for t, l in zip(list(X_train), list(y_train))]
        valids = [(t, l) for t, l in zip(list(X_val), list(y_val))]
        train_generator = data_generator(trains, batch_size)
        valid_generator = data_generator(valids, batch_size)

        model = build_model()
        evaluator = Evaluator()
        model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            callbacks=[evaluator]
        )
        fw = open('./output/{}.expr'.format(expr), 'w')
        fw.write(json.dumps(vars(args), indent=2) + '\n')
        fw.write(json.dumps(evaluator.metric_history, indent=2) + '\n')
        fw.close()

    if args.predict:
        expr = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        test_data = load_test(test_data_path)
        tests = [(t[2], 0) for t in test_data]
        test_generator = data_generator(tests, batch_size)

        model = build_model()
        model.load_weights('best_model.weights')
        y_hat = predict(test_generator)
        with open('.{}_submission.csv'.format(expr), 'w') as fw:
            for d, l in zip(test_data, y_hat):
                # if int(l) == 1:
                    # fw.write('{}\t{}\n'.format(d, l))
                fw.write("\t".join(d) + '\t{}\n'.format(l))


if __name__ == '__main__':
    main()