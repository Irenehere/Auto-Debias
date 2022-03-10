import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertForMaskedLM

def load_word_list(f_path):
    lst = []
    with open(f_path,'r') as f:
        line = f.readline()
        while line:
            lst.append(line.strip())
            line = f.readline()
    return lst

def load_wiki_word_list(f_path):
    vocab = []
    with open(f_path,"r")as f:
        line = f.readline()
        while line:
            vocab.append(line.strip().split()[0])
            line = f.readline()
    return vocab

class JSD(nn.Module):
    def __init__(self,reduction='batchmean'):
        super(JSD, self).__init__()
        self.reduction = reduction

    def forward(self, net_1_logits, net_2_logits):
        net_1_probs = F.softmax(net_1_logits, dim=1)
        net_2_probs= F.softmax(net_2_logits, dim=1)

        total_m = 0.5 * (net_1_probs + net_2_probs)
        loss = 0.0
        loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), total_m, reduction=self.reduction) 
        loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), total_m, reduction=self.reduction) 
     
        return (0.5 * loss) 
    
def clean_vocab(vocab):
    new_vocab = []
    for v in vocab:
        if (v[0] not in ['#','[','.','0','1','2','3','4','5','6','7','8','9']) and len(v)>1:
            new_vocab.append(v)
    return new_vocab


def clean_word_list2(tar1_words_,tar2_words_,tokenizer):
    tar1_words = []
    tar2_words = []
    for i in range(len(tar1_words_)):
        if tokenizer.convert_tokens_to_ids(tar1_words_[i])!=tokenizer.unk_token_id and tokenizer.convert_tokens_to_ids(tar2_words_[i])!=tokenizer.unk_token_id:
            tar1_words.append(tar1_words_[i])
            tar2_words.append(tar2_words_[i])
    return tar1_words, tar2_words

def clean_word_list(vocabs,tokenizer):
    vocab_list = []
    for i in range(len(vocabs)):
        if tokenizer.convert_tokens_to_ids(vocabs[i])!=tokenizer.unk_token_id:
            vocab_list.append(vocabs[i])
    return vocab_list