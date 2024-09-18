

import random
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import *



from create_dataset import MOSI, MOSEI, UR_FUNNY, PAD, UNK

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")


vocab_file = '/home/s22xjq/SATI/model/vocab.json'
merges_file = '/home/s22xjq/SATI/model/merges.txt'
roberta_tokenizer = RobertaTokenizer(vocab_file, merges_file)


class MSADataset(Dataset):
    def __init__(self, config):

        ## Fetch dataset
        if "mosi" in str(config.data_dir).lower():
            dataset = MOSI(config)
        elif "mosei" in str(config.data_dir).lower():
            dataset = MOSEI(config)
        elif "ur_funny" in str(config.data_dir).lower():
            dataset = UR_FUNNY(config)
        else:
            print("Dataset not defined correctly")
            exit()
        
        self.data, self.word2id, self.pretrained_emb = dataset.get_data(config.mode)
        #print(self.data)
        self.len = len(self.data)

        config.visual_size = self.data[0][0][1].shape[1]
        config.acoustic_size = self.data[0][0][2].shape[1]

        config.word2id = self.word2id
        config.pretrained_emb = self.pretrained_emb


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len



def get_loader(config, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    dataset = MSADataset(config)
    
    #print(config.mode)
    config.data_len = len(dataset)


    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
        
        # get the data out of the batch - use pad sequence util functions from PyTorch to pad things


        labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
        sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)#[44, 64]
        #print(sentences.shape)
        visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])  #[44, 64, 47]
        #print(visual)
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch]) #[44, 64, 47]
        #print(sentences.shape, visual.shape)

        ## BERT-based features input prep

        SENT_LEN = sentences.size(0)
        # Create bert indices using tokenizer
        roberta_details = []
        for sample in batch:
            text = " ".join(sample[0][3])  # 将文本合并成一个字符串
            encoded_roberta_sent = roberta_tokenizer.encode_plus(
                text, 
                max_length=SENT_LEN,  # RoBERTa不需要+2，因为它只需要[CLS]和[SEP]标记
                add_special_tokens=True, 
                padding='max_length',  # 填充到最大长度
                truncation=True,       # 截断超过最大长度的部分
                return_tensors='pt'    # 返回PyTorch tensors（如果你使用的是PyTorch）
            )
            roberta_details.append(encoded_roberta_sent)

            bert_sentences = torch.LongTensor([sample["input_ids"].squeeze(0).tolist() for sample in roberta_details])
            bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"].squeeze(0).tolist() for sample in roberta_details])
            bert_sentence_types = torch.randn(0)
        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])

        return sentences, visual, acoustic, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask


    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)

    return data_loader
