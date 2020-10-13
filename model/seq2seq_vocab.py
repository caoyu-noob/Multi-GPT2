import os
import pickle

class Seq2seqTokenizer:
    def __init__(self):
        self.word2idx = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3, '<talker1_bos>': 4, '<talker2_bos>': 5}
        self.idx2word = {0: "<unk>", 1: "<pad>", 2: "<bos>", 3: "<eos>", 4: "<talker1_bos>", 5: "<talker2_bos>"}
        self.n_words = 6
        self.all_special_ids = [0, 1, 2, 3, 4, 5]
        self.pad_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self.talker1_bos_id = 4
        self.talker2_bos_id = 5

    def tokenize(self, str):
        res = str.strip().split(' ')
        res = [x.lower() for x in res]
        return res

    def encode(self, tokenized_str):
        res = []
        for token in tokenized_str:
            if self.word2idx.__contains__(token):
                res.append(self.word2idx[token])
        return res

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        res = []
        for id in ids:
            if skip_special_tokens and id in self.all_special_ids:
                continue
            res.append(self.idx2word[id])
        text = ' '.join(res)
        return text

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if not self.word2idx.__contains__(word):
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1

class Seq2seqVocab:
    def __init__(self, train_dataset_path, valid_dataset_path, test_dataset_path, vocab_path, data_type='persona'):
        if (os.path.exists(vocab_path)):
            with open(vocab_path, 'rb') as f:
                cached_data = pickle.load(f)
            self.vocab = cached_data[0]
            self.all_data = cached_data[1]
        else:
            self.vocab = Seq2seqTokenizer()
            if data_type == 'persona':
                self.all_data = self.parse_data(train_dataset_path, valid_dataset_path, test_dataset_path)
            elif data_type == 'emoji':
                self.all_data = self.parse_data_emoji(train_dataset_path, valid_dataset_path, test_dataset_path)
            self.parse_vocab(self.all_data, self.vocab)
            with open(vocab_path, 'wb') as f:
                pickle.dump([self.vocab, self.all_data], f)

    def parse_data(self, train_dataset_path, valid_dataset_path, test_dataset_path):
        subsets = [train_dataset_path, valid_dataset_path, test_dataset_path]
        all_data = []
        for subset in subsets:
            data = []
            with open(subset, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    space_idx = line.find(' ')
                    if space_idx == -1:
                        dialog_idx = int(line)
                    else:
                        dialog_idx = int(line[:space_idx])

                    if int(dialog_idx) == 1:
                        data.append({'persona_info': [], 'dialog': [], 'candidates': []})

                    dialog_line = line[space_idx + 1:].split('\t')
                    dialog_line = [l.strip() for l in dialog_line]

                    if dialog_line[0].startswith('your persona:'):
                        persona_info = dialog_line[0].replace('your persona: ', '')
                        persona_info = persona_info.replace('.', ' .')
                        data[-1]['persona_info'].append(persona_info)

                    elif len(dialog_line) > 1:
                        data[-1]['dialog'].append(dialog_line[0])
                        data[-1]['dialog'].append(dialog_line[1])
                    if len(dialog_line) == 4:
                        data[-1]['candidates'].append(dialog_line[3].split('|')[:-1])

            all_data.append(data)
        return all_data

    def parse_data_emoji(self, train_dataset_path, valid_dataset_path, test_dataset_path):
        subsets = [train_dataset_path, valid_dataset_path, test_dataset_path]
        all_data = []
        for subset in subsets:
            data = []
            with open(subset, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    items = line.split('\t')
                    data.append({'persona_info': [], 'dialog': [], 'candidates': []})
                    data[-1]['persona_info'].append(items[0])
                    data[-1]['dialog'].append(items[1])
                    data[-1]['dialog'].append(items[2])
            all_data.append(data)
        return all_data

    def parse_vocab(self, all_data, vocab):
        for data in all_data:
            for p in data:
                for s in p['persona_info']:
                    vocab.index_words(s)
                for s in p['dialog']:
                    vocab.index_words(s)
                for c in p['candidates']:
                    for s in c:
                        vocab.index_words(s)