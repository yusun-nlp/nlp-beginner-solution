from data_helper import read_category, read_vocab
from model import TextRNN
import torch
import tensorflow.keras as kr


class RnnModel:
    def __init__(self):
        self.categories, self.cat_id = read_category()
        self.words, self.word_id = read_vocab('./data/cnews.vocab.txt')
        self.model = TextRNN()
        self.model.load_state_dict(torch.load('model_params.pkl'))

    def predict(self, input):
        data = [self.word_id[x] for x in input if x in self.word_id]
        data = kr.preprocessing.sequence.pad_sequences([data], 600)
        data = torch.LongTensor(data)
        Y_pred = self.model(data)
        class_index = torch.argmax(Y_pred[0]).item()
        return self.categories[class_index]


if __name__ == '__main__':
    model = RnnModel()
    X_test = ['三星ST550以全新的拍摄方式超越了以往任何一款数码相机',
              '热火vs骑士前瞻：皇帝回乡二番战 东部次席唾手可得新浪体育讯北京时间3月30日7:00']
    for i in X_test:
        print(i, ':', model.predict(i))
