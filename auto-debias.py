import argparse
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from utils import *
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, get_scheduler
from transformers import BertTokenizer,BertForPreTraining 
from transformers import RobertaTokenizer,RobertaForMaskedLM,RobertaModel
from transformers import AlbertTokenizer, AlbertForPreTraining

parser = argparse.ArgumentParser()

parser.add_argument(
    "--debias_type",
    default='gender',
    type=str,
    choices=['gender','race'],
    help="Choose from ['gender','race']",
)

parser.add_argument(
    "--model_name_or_path",
    default="bert-base-uncased",
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models",
)

parser.add_argument(
    "--model_type",
    default="bert",
    type=str,
    help="choose from ['bert','roberta','albert']",
)

parser.add_argument(
    "--data_path",
    default="data/",
    type=str,
    help="data path to put the taget/attribute word list",
)


parser.add_argument(
    "--prompts_file",
    default="",
    type=str,
    help="the name of the file that stores the prompts, by default it is under the data_path",
)

parser.add_argument(
    "--batch_size",
    default=32,
    type=int,
    help="batch size in auto-debias fine-tuning",
)

parser.add_argument(
    "--lr",
    default=5e-6,
    type=float,
    help="learning rate in auto-debias fine-tuning",
)

parser.add_argument(
    "--epochs",
    default=1,
    type=int,
    help="number of epochs in auto-debias fine-tuning",
)

parser.add_argument(
    "--finetuning_vocab_file",
    default=None,
    type=str,
    help="vocabulary to be fine-tuned in auto-debias fine-tuning, if None, tune the whole vocabulary.",
)

parser.add_argument(
    "--tune_pooling_layer",
    default=False,
    type=str,
    help="whether to tune the pooling layer with the auxiliary loss",
)


def get_tokenized_prompt(prompts,tar1_words,tar2_words,tokenizer):
    tar1_sen = []
    tar2_sen = []
    for i in range(len(prompts)):
        for j in range(len(tar1_words)):
            tar1_sen.append(tar1_words[j]+" "+prompts[i]+" "+tokenizer.mask_token+".")
            tar2_sen.append(tar2_words[j]+" "+prompts[i]+" "+tokenizer.mask_token+".")
    tar1_tokenized = tokenizer(tar1_sen,padding=True, truncation=True, return_tensors="pt")
    tar2_tokenized = tokenizer(tar2_sen,padding=True, truncation=True, return_tensors="pt")
    tar1_mask_index = np.where(tar1_tokenized['input_ids'].numpy()==tokenizer.mask_token_id)[1]
    tar2_mask_index = np.where(tar2_tokenized['input_ids'].numpy()==tokenizer.mask_token_id)[1]
    print(tar1_tokenized['input_ids'].shape)    
    return tar1_tokenized,tar2_tokenized, tar1_mask_index, tar2_mask_index

def send_to_cuda(tar1_tokenized,tar2_tokenized):
    for key in tar1_tokenized.keys():
        tar1_tokenized[key] = tar1_tokenized[key].cuda()
        tar2_tokenized[key] = tar2_tokenized[key].cuda()
    return tar1_tokenized,tar2_tokenized



if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        model = BertForPreTraining.from_pretrained(args.model_name_or_path)
    elif args.model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        model = RobertaForMaskedLM.from_pretrained(args.model_name_or_path)
        new_roberta= RobertaModel.from_pretrained(args.model_name_or_path) #make the add_pooling_layer=True
        model.roberta = new_roberta
    elif args.model_type == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained(args.model_name_or_path)
        model = AlbertForPreTraining.from_pretrained(args.model_name_or_path)
    else:
        raise NotImplementedError("not implemented!")
    model.train()
    model.cuda()
    searched_prompts = load_word_list(args.data_path+args.prompts_file)
    if args.debias_type == 'gender':
        male_words_ = load_word_list(args.data_path+"male.txt")
        female_words_ = load_word_list(args.data_path+"female.txt")
        tar1_words, tar2_words = clean_word_list2(male_words_, female_words_,tokenizer)   #remove the OOV words
        tar1_tokenized,tar2_tokenized,tar1_mask_index,tar2_mask_index = get_tokenized_prompt(searched_prompts, tar1_words, tar2_words, tokenizer)
        tar1_tokenized,tar2_tokenized =send_to_cuda(tar1_tokenized,tar2_tokenized)
    elif args.debias_type=='race':
        race1_words_ = load_word_list(args.data_path+"race1.txt")
        race2_words_ = load_word_list(args.data_path+"race2.txt")
        tar1_words, tar2_words = clean_word_list2(race1_words_, race2_words_,tokenizer)
        tar1_tokenized,tar2_tokenized,tar1_mask_index,tar2_mask_index = get_tokenized_prompt(searched_prompts, tar1_words, tar2_words, tokenizer)
        tar1_tokenized,tar2_tokenized =send_to_cuda(tar1_tokenized,tar2_tokenized)

    if args.finetuning_vocab_file:
        finetuning_vocab_ = load_word_list(args.data_path+args.finetuning_vocab_file)
        finetuning_vocab = tokenizer.convert_tokens_to_ids(finetuning_vocab_)
    
    jsd_model = JSD()
    
    assert tar1_tokenized['input_ids'].shape[0] == tar2_tokenized['input_ids'].shape[0]
    data_len = tar1_tokenized['input_ids'].shape[0]

    idx_ds = DataLoader([i for i in range(data_len)], batch_size = args.batch_size, shuffle=True,drop_last=True)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    for i in range(1,args.epochs+1):  
        print("epoch",i)

        # load data
        for idx in idx_ds:
            tar1_inputs={}
            tar2_inputs={}            
            for key in tar1_tokenized.keys():
                tar1_inputs[key]=tar1_tokenized[key][idx]
                tar2_inputs[key]=tar2_tokenized[key][idx]         
            tar1_mask = tar1_mask_index[idx]
            tar2_mask = tar2_mask_index[idx]

            optimizer.zero_grad()
 
            tar1_predictions = model(**tar1_inputs)
            tar2_predictions = model(**tar2_inputs)

            if args.finetuning_vocab_file:   
                tar1_predictions_logits = tar1_predictions.prediction_logits[torch.arange(tar1_predictions.prediction_logits.size(0)), tar1_mask][:,finetuning_vocab]
                tar2_predictions_logits = tar2_predictions.prediction_logits[torch.arange(tar2_predictions.prediction_logits.size(0)), tar2_mask][:, finetuning_vocab]
            else:
                tar1_predictions_logits = tar1_predictions.prediction_logits[torch.arange(tar1_predictions.prediction_logits.size(0)), tar1_mask]
                tar2_predictions_logits = tar2_predictions.prediction_logits[torch.arange(tar2_predictions.prediction_logits.size(0)), tar2_mask]

            jsd_loss = jsd_model(tar1_predictions_logits,tar2_predictions_logits)
            loss =jsd_loss
            
            if args.tune_pooling_layer:
                if args.model_type == 'bert':
                    tar1_embedding = model.bert(**tar1_inputs).pooler_output
                    tar2_embedding = model.bert(**tar2_inputs).pooler_output
                elif args.model_type == 'roberta':
                    tar1_embedding = model.roberta(**tar1_inputs).pooler_output
                    tar2_embedding = model.roberta(**tar2_inputs).pooler_output
                elif args.model_type == 'albert':
                    tar1_embedding = model.albert(**tar1_inputs).pooler_output
                    tar2_embedding = model.albert(**tar2_inputs).pooler_output   
                embed_dist = 1-F.cosine_similarity(tar1_embedding,tar2_embedding,dim=1)
                embed_dist = torch.mean(embed_dist)
                loss =jsd_loss+0.1*torch.mean(embed_dist)
             
            loss.backward()  
            optimizer.step()
            optimizer.zero_grad()
            print('jsd loss {}'.format(jsd_loss))
        model.save_pretrained('model/debiased_model_{}_{}'.format(args.model_name_or_path, args.debias_type))
        tokenizer.save_pretrained('model/debiased_model_{}_{}'.format(args.model_name_or_path, args.debias_type))
