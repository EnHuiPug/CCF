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
from sklearn.model_selection import train_test_split 
from keras.utils import to_categorical
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

tqdm.pandas(desc='demo')

def main():
    parser = argparse.ArgumentParser(description="Run text classifier")
    parser.add_argument('--train', default=False, type=bool)
    parser.add_argument('--predict', default=False, type=bool)
    parser.add_argument('--maxlen', default=512, type=int)
    parser.add_argument('--epochs',default=10, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--train_data', default='./data/labeled_data.csv')
    parser.add_argument('--supply_train_data', default='./data/labeled_supply_data.csv')
    parser.add_argument('--add_supply_data', default=False, type=bool)
    parser.add_argument('--test_data', default='./data/test_data.csv')
    parser.add_argument('--pretrain_model', default='/home/david/pretrain_model/google_bert/chinese_L-12_H-768_A-12')
    parser.add_argument('--task', default='category')
    parser.add_argument('--train_data_split_ratio', default=0.2, type=float)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--ensamble', default=False, type=bool)
    parser.add_argument('--patience', default=5)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    args = parser.parse_args()
    print(args)

    if args.ensamble:
        result = {}
        files = os.listdir('./output')
        for f in files:
            if f.endswith('_submission.csv'):
                with open(os.path.join('./output', f), 'r') as fr:
                    for line in fr.readlines():
                        if line.startswith('id'):
                            continue
                        if line.strip().split(',')[0] not in result:
                            result[line.strip().split(',')[0]] = set()
                            if line.strip().split(',')[1] != '0':
                                result[line.strip().split(',')[0]].add(line)
        # print(result)
        missing_domain = ['游戏,低风险', '体育,可公开', '娱乐,可公开']
        with open('./output/merged_submission.csv', 'w') as fw:
            fw.write('id,class_label,rank_label\n')
            for k,v in result.items():
                if len(v) != 0:
                    fw.write(random.choice(list(v)))
                else:
                    fw.write('{},{}\n'.format(k, random.choice(missing_domain)))
        


    maxlen = args.maxlen
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    pretrain_model = args.pretrain_model
    config_path = os.path.join(pretrain_model, 'bert_config.json')
    checkpoint_path = os.path.join(pretrain_model, 'bert_model.ckpt')
    dict_path = os.path.join(pretrain_model, 'vocab.txt')
    # 建立分词器
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    # data
    train_data_path = args.train_data
    test_data_path = args.test_data

    maps = {
        '财经':'高风险',
        '时政':'高风险',
        '房产':'中风险',
        '科技':'中风险',
        '教育':'低风险',
        '时尚':'低风险',
        '游戏':'低风险',
        '家居':'可公开',
        '体育':'可公开',
        '娱乐':'可公开',
    }
    if not args.add_supply_data:
        label2id = {'时政': 0, '房产': 1, '财经': 2, '科技': 3, '时尚': 4, '教育': 5, '家居': 6}
        id2label = {0:'时政', 1:'房产', 2:'财经', 3:'科技', 4:'时尚', 5:'教育', 6:'家居'}
    else:
        label2id = {'时政': 0, '房产': 1, '财经': 2, '科技': 3, '时尚': 4, '教育': 5, '家居': 6, '游戏':7, '娱乐':8, '体育':9}
        id2label = {0:'时政', 1:'房产', 2:'财经', 3:'科技', 4:'时尚', 5:'教育', 6:'家居', 7:'游戏', 8:'娱乐', 9:'体育'}

    def load_train(filename, task):
        df = pd.read_csv(filename)
        if task == 'category':
            df['label'] = df['class_label'].progress_map(lambda x: label2id[x])
        text = []
        label = []
        for t, l in zip(df['content'], df['label']):
            text.append(t)
            label.append(l)
        return text, label # [text1, ...], [label1, ...]

    def load_test(filename):
        df = pd.read_csv(filename)
        text = []
        for t in df['content']:
            text.append(t)
        return text

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
            units=len(label2id) if args.task == 'category' else 1,
            activation='softmax' if args.task == 'category' else 'sigmoid',
            kernel_initializer=bert.initializer
        )(output)

        model = keras.models.Model(bert.model.input, output)
        model.summary()

        # AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

        model.compile(
            loss='sparse_categorical_crossentropy' if args.task == 'category' else 'binary_crossentropy',
            optimizer=Adam(learning_rate),  # 用足够小的学习率
            # optimizer=AdamLR(learning_rate=1e-4, lr_schedule={
            #     1000: 1,
            #     2000: 0.1
            # }),
            metrics=['accuracy'],
        )
        return model


    def evaluate(data):
        f1, precision, recall, accuracy = 0., 0., 0., 0.,
        y, y_ = [], []
        
        for x_true, y_true in data:
            if args.task == 'category': 
                y_pred = model.predict(x_true).argmax(axis=1)
            else:
                y_p = model.predict(x_true)
                y_pred = []
                for i in y_p:
                    if i >= args.threshold:
                        y_pred.append(1)
                    else:
                        y_pred.append(0)
            y.extend(y_true)
            y_.extend(y_pred)
        if args.task == 'category':
            f1 = f1_score(y, y_, average='macro')
            precision = precision_score(y, y_, average='macro')
            recall = recall_score(y, y_, average='macro')
            accuracy = accuracy_score(y, y_)
        else:
            f1 = f1_score(y, y_)
            precision = precision_score(y, y_)
            recall = recall_score(y, y_)
            accuracy = accuracy_score(y, y_)
        return f1, precision, recall, accuracy

    def predict(data):
        y_hat = []
        for x_true, _ in tqdm(data):
            if args.task == 'category': 
                y_pred = model.predict(x_true).argmax(axis=1)
            else:
                y_p = model.predict(x_true)
                y_pred = []
                for i in y_p:
                    if i >= args.threshold:
                        y_pred.append(1)
                    else:
                        y_pred.append(0)
            for i in y_pred:
                y_hat.append(i)
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

        def on_epoch_end(self, epoch, logs=None):
            f1, precision, recall, accuracy = evaluate(valid_generator)
            self.metric_history[epoch] = {'f1':f1, 'precision':precision, 'recall':recall, 'accuracy':accuracy}
            if f1 > self.best_val_f1:
                self.wait = 0
                self.best_val_f1 = f1
                self.best_epoch = epoch
                self.metric_history['best'] = {'best_val_f1': self.best_val_f1, 'best_epoch':self.best_epoch}
                self.model.save_weights('best_model.weights')
            print(
                u'val_f1: %.5f, 'u'val_precision: %.5f, 'u'val_recall: %.5f, 'u'val_accuracy: %.5f, best_val_f1: %.5f\n' %
                (f1, precision, recall, accuracy, self.best_val_f1)
            )
            self.wait += 1
            if self.wait > args.patience:
                self.model.stop_training = True


    assert args.train or args.predict == True

    expr = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.train:
        print('Training process')

        # 读取数据
        if not args.add_supply_data:
            train_text, train_label = load_train(train_data_path, args.task)
        else:
            train_text, train_label = load_train(train_data_path, args.task)
            supply_train_text, supply_train_label = load_train(args.supply_train_data, args.task)
            train_text = train_text + supply_train_text
            train_label = train_label + supply_train_label
        
        assert len(train_text) == len(train_label)
        print('Raw labeled data number: ', len(train_text))

        
        X_train, X_val, y_train, y_val = train_test_split(train_text, train_label, test_size=args.train_data_split_ratio, random_state=42)

        trains = [(t,l) for t, l in zip(list(X_train),list(y_train))]
        valids = [(t,l) for t, l in zip(list(X_val),list(y_val))]
        
        train_generator = data_generator(trains,batch_size)
        valid_generator = data_generator(valids,batch_size)
       

        evaluator = Evaluator()
        model = build_model()
        model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            callbacks=[evaluator]
        )
        fw = open('./output/{}_infos.expr'.format(expr), 'w')
        fw.write(json.dumps(vars(args), indent=2) + '\n')
        fw.write(json.dumps(evaluator.metric_history, indent=2)+'\n')
        fw.close()

    if args.predict:
        test_text = load_test(test_data_path)
        tests = [(t,0) for t in test_text]
        test_generator = data_generator(tests,batch_size)
        model = build_model()
        
        if args.task == 'category':
            model.load_weights('best_model.weights')
            y_hat = predict(test_generator)
            with open('./output/{}_submission.csv'.format(expr), 'w') as fw:
                fw.write('id,class_label,rank_label\n')
                for i, ele in enumerate(y_hat):
                    try: 
                        fw.write('{},{},{}\n'.format(i, id2label[ele], maps[id2label[ele]]))
                    except:
                        print(ele)
        else:
            model.load_weights('best_model.weights')
            y_hat = predict(test_generator)
            base = basename(args.train_data).replace('_binary_data.csv','')
            with open('./output/{}_submission.csv'.format(base), 'w') as fw:
                fw.write('id,class_label,rank_label\n')
                for i, ele in enumerate(y_hat):
                    try: 
                        fw.write('{},{},{}\n'.format(i, 
                                                    base if ele ==1 else 0, 
                                                    maps[base] if ele ==1 else 0))
                    except:
                        print(ele)


if __name__ == '__main__':

    main()