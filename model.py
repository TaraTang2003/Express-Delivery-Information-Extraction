# -*- coding = utf-8 -*-
# @Time : 2023/9/24 11:01
# @Author : TX
# @File : model.py
# @Software : PyCharm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch.optim as optim
from torchcrf import CRF

# Dataset
class NerDataset(Dataset):
    def __init__(self, datafile):
        self.datafile = datafile
        self.data = self.read_data()
        self.label_vocab = self.load_dict('./conf/tag.dic')
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.vocab_size = self.tokenizer.vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words = self.data[idx]['word']
        labels = self.data[idx]['label']

        inputs = self.tokenizer(
            words, is_split_into_words=True, return_offsets_mapping=True,
            max_length=self.get_label_size(), padding='max_length', truncation=True
        )
        word_ids = inputs['input_ids']
        attention_mask = torch.tensor(inputs['attention_mask'])
        bool_attention_mask = attention_mask == 1

        padded_labels = labels + ['O'] * (self.get_label_size() - len(labels))
        label_ids = [self.label_vocab.get(label, 'O') for label in padded_labels]

        return {
            'word_ids': torch.tensor(word_ids),
            'attention_mask': bool_attention_mask,
            'label_ids': torch.tensor(label_ids)
        }

    def load_dict(self, dict_path):
        vocab = {}
        i = 0
        for line in open(dict_path, 'r', encoding='utf-8'):
            key = line.strip('\n')
            vocab[key] = i
            i += 1
        return vocab

    def read_data(self):
        data = []
        with open(self.datafile, 'r', encoding='utf-8') as fp:
            next(fp)
            for line in fp.readlines():
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                data.append({'word':words, 'label':labels})
        return data

    def get_label_size(self):
        size = 0
        for i in self.data:
            size = max(size, len(i['label']))
        return size

class BiGRU_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, labels_dim, num_layers, vocab_size):
        super(BiGRU_CRF, self).__init__()

        # Word Embedding (num_embeddings, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # BiGRU
        self.bi_gru = nn.GRU(embedding_dim,
                             hidden_dim,
                             num_layers=num_layers,
                             bidirectional=True)

        # Encode
        self.fc = nn.Linear(hidden_dim * 2, labels_dim)
        # CRF
        self.crf = CRF(labels_dim,  batch_first=True)

    def forward(self, text):
        # text: [batch_size, seq_len]
        embedded = self.embedding(text)
        # embedded: [batch_size, seq_len, embedding_dim]

        output, _ = self.bi_gru(embedded)
        # output: [batch_size, seq_len, hidden_dim * 2]

        predictions = self.fc(output)
        # predictions: [batch_size, seq_len, labels_dim]

        return predictions

    def forward_with_crf(self, word_ids, mask, label_ids):
        tag_scores = self.forward(word_ids)
        loss = self.crf(tag_scores, label_ids, mask) * (-1)
        return tag_scores, loss

if __name__ == "__main__":
    # load dataset
    train_ds = NerDataset('./express_ner/train.txt')
    test_ds = NerDataset('./express_ner/test.txt')
    dev_ds = NerDataset('./express_ner/dev.txt')

    # DataLoader
    batch_size = 32
    shuffle = True
    drop_last = True
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    dev_loader = DataLoader(
        dataset=dev_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    for i, data in enumerate(dev_loader):
        batch_data = data['word_ids']
        attention_mask = data['attention_mask']
        batch_label = data['label_ids']
        break

    # Init Model
    embedding_dim = 300
    hidden_dim = 300
    labels_dim = train_ds.get_label_size()
    num_layers = 1
    vocab_size = train_ds.vocab_size

    # Build Model
    model = BiGRU_CRF(embedding_dim, hidden_dim, labels_dim, num_layers, vocab_size)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    # Train Model
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader):
            batch_data = data['word_ids']
            attention_mask = data['attention_mask']
            batch_label = data['label_ids']
            optimizer.zero_grad()
            tag_scores, loss = model.forward_with_crf(batch_data, attention_mask, batch_label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss = float(train_loss) / len(train_loader)
        print("Epoch:", epoch, ";", "Loss", loss.item())

    # Evaluate Model
    model.eval()

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for i, data in enumerate(dev_loader):
            batch_data = data['word_ids']
            attention_mask = data['attention_mask']
            batch_label = data['label_ids']

            relabel_rows = torch.unbind(batch_label, dim=0)
            mask_rows = torch.unbind(attention_mask, dim=0)

            tag_scores = model.forward(batch_data)
            prelable_rows = model.crf.decode(tag_scores, mask=attention_mask)

            for i in range(len(prelable_rows)):
                print(relabel_rows[i][mask_rows[i]].tolist())
                print(prelable_rows[i])
                print()
                if relabel_rows[i][mask_rows[i]].tolist() == prelable_rows[i]:
                    total_correct += 1
                total_samples += 1
        score = total_correct / total_samples