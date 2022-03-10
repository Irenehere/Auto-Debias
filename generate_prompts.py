'''beam search for prompts that can probe the bias'''

import time
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from transformers import BertTokenizer,BertForMaskedLM
from transformers import RobertaTokenizer,RobertaForMaskedLM
from transformers import AlbertTokenizer,AlbertForMaskedLM

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
    help="Choose from ['bert','roberta','albert']",
)

parser.add_argument(
    "--vocab_file",
    default='data/wiki_words_5000.txt',
    type=str,
    help="Path to the file that stores the vocabulary of the prompts",
)

parser.add_argument(
    "--BS",
    default=1000,
    type=int,
    help="batch size of the data fed into the model",
)

parser.add_argument(
    "--PL",
    default=5,
    type=int,
    help="maximun length of the generated prompts",
)

parser.add_argument(
    "--K",
    default=100,
    type=int,
    help="top K prompts to be selected in the beam search",
)



def send_to_cuda(tar1_tokenized,tar2_tokenized):
    for key in tar1_tokenized.keys():
        tar1_tokenized[key] = tar1_tokenized[key].cuda()
        tar2_tokenized[key] = tar2_tokenized[key].cuda()
    return tar1_tokenized,tar2_tokenized


def get_tokenized_ith_prompt(prompts,tar1_word,tar2_word,tokenizer):
    tar1_sen_i = []
    tar2_sen_i = []
    for i in range(len(prompts)):
        tar1_sen_i.append(tar1_word+" "+prompts[i]+" "+tokenizer.mask_token)
        tar2_sen_i.append(tar2_word+" "+prompts[i]+" "+tokenizer.mask_token)
    tar1_tokenized = tokenizer(tar1_sen_i,padding=True, truncation=True, return_tensors="pt")
    tar2_tokenized = tokenizer(tar2_sen_i,padding=True, truncation=True, return_tensors="pt")
    tar1_mask_index = np.where(tar1_tokenized['input_ids'].numpy()==tokenizer.mask_token_id)[1]
    tar2_mask_index = np.where(tar2_tokenized['input_ids'].numpy()==tokenizer.mask_token_id)[1]
    print([tokenizer.convert_ids_to_tokens(int(id_)) for id_ in tar1_tokenized['input_ids'][0]])
    assert tar1_mask_index.shape[0]==tar1_tokenized['input_ids'].shape[0]
    return tar1_tokenized,tar2_tokenized,tar1_mask_index,tar2_mask_index


def run_model(model,inputs,mask_index,ster_words):
    predictions = model(**inputs)
    predictions_logits = predictions.logits[np.arange(inputs['input_ids'].size(0)), mask_index][:,ster_words]
    return predictions_logits


def get_JSD(tar1_tokenized,tar2_tokenized,tar1_mask_index,tar2_mask_index,model,ster_words):
    jsd_list=[]
    tar1_tokenized,tar2_tokenized = send_to_cuda(tar1_tokenized,tar2_tokenized)
    for k in range(tar1_tokenized['input_ids'].shape[0]//args.BS + 1):
        tar1_inputs={}
        tar2_inputs={}
        try:  
            for key in tar1_tokenized.keys():
                tar1_inputs[key]=tar1_tokenized[key][args.BS*k:args.BS*(k+1)]
                tar2_inputs[key]=tar2_tokenized[key][args.BS*k:args.BS*(k+1)]
                            
            tar1_local_mask_index = tar1_mask_index[args.BS*k:args.BS*(k+1)]
            tar2_local_mask_index = tar2_mask_index[args.BS*k:args.BS*(k+1)]
        except IndexError:
            for key in tar1_tokenized.keys():
                tar1_inputs[key]=tar1_tokenized[key][args.BS*(k+1):]
                tar2_inputs[key]=tar2_tokenized[key][args.BS*(k+1):]
                            
            tar1_local_mask_index = tar1_mask_index[args.BS*(k+1):]
            tar2_local_mask_index = tar2_mask_index[args.BS*(k+1):]

        #stopped here
        tar1_predictions_logits = run_model(model,tar1_inputs,tar1_local_mask_index,ster_words)
        tar2_predictions_logits = run_model(model,tar2_inputs,tar2_local_mask_index,ster_words)
 
        jsd = jsd_model(tar1_predictions_logits,tar2_predictions_logits)
        jsd_np = jsd.detach().cpu().numpy()
        jsd_np = np.sum(jsd_np,axis=1)
        jsd_list += list(jsd_np)
        del tar1_predictions_logits, tar2_predictions_logits, jsd
    return jsd_list 


def get_prompt_jsd(tar1_words, tar2_words, prompts, model, ster_words):
    jsd_word_list = []
    assert len(tar1_words)==len(tar2_words)
    for i in range(len(tar1_words)): 
        tar1_tokenized_i,tar2_tokenized_i,tar1_mask_index_i,tar2_mask_index_i = get_tokenized_ith_prompt(prompts,tar1_words[i],tar2_words[i],tokenizer)
        print("tokenized input shape",tar1_tokenized_i['input_ids'].shape)        
        jsd_list = get_JSD(tar1_tokenized_i,tar2_tokenized_i, tar1_mask_index_i, tar2_mask_index_i, model, ster_words)
        jsd_word_list.append(jsd_list)
        print("got the jsd for the word",tar1_words[i])
    jsd_word_list = np.array(jsd_word_list)
    print("jsd for every prompt, every word has shape",jsd_word_list.shape)
    assert jsd_word_list.shape == (len(tar1_words),len(prompts))    
    return np.mean(jsd_word_list, axis=0)


if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        model = BertForMaskedLM.from_pretrained(args.model_name_or_path)
    elif args.model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        model = RobertaForMaskedLM.from_pretrained(args.model_name_or_path)
    elif args.model_type == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained(args.model_name_or_path)
        model = AlbertForMaskedLM.from_pretrained(args.model_name_or_path)
    else:
        raise NotImplementedError("not implemented!")
    model = torch.nn.DataParallel(model)
    model.eval()
    model.cuda()
    
    jsd_model = JSD(reduction='none')

    if args.debias_type == 'gender':
        male_words=['fathers','actor','prince','men','gentlemen','sir','brother','his','king','husband','dad','males','sir','him','boyfriend','he','hero',                'kings','brothers','son','sons','himself','gentleman','his','father','male','man','grandpa','boy','grandfather']
        female_words=[ 'mothers','actress','princess','women','ladies','madam','sister','her','queen','wife','mom','females','miss','her','girlfriend',                  'she','heroine','queens','sisters','daughter','daughters','herself','lady','hers','mother','female','woman','grandma','girl','grandmother']

    elif args.debias_type=='race':
        race1 = ["black","african","black","africa","africa","africa","black people","african people","black people","the africa"]
        race2 = ["caucasian","caucasian","white","america","america","europe","caucasian people","caucasian people","white people","the america"]

    ster_words_ = clean_word_list(load_word_list("data/stereotype.txt"),tokenizer)
    ster_words = tokenizer.convert_tokens_to_ids(ster_words_)   #stereotype words   

    vocab = load_wiki_word_list(args.vocab_file)           
    vocab = clean_word_list(vocab,tokenizer)     #vocabulary in prompts
    # print(vocab)

    current_prompts = vocab

    f=open('data/prompts_{}_{}'.format(args.model_name_or_path,args.debias_type),'w')
    for m in range(args.PL):
        if args.debias_type == 'gender':
            current_prompts_jsd = get_prompt_jsd(male_words, female_words,current_prompts, model,ster_words)
        elif args.debias_type=='race':
            current_prompts_jsd = get_prompt_jsd(race1, race2,current_prompts, model,ster_words)
        top_k_prompts = np.array(current_prompts)[np.argsort(current_prompts_jsd)[::-1][:args.K]]
        print(top_k_prompts)
        for p in top_k_prompts:
            f.write(p)
            f.write("\n")
        new_prompts = []
        for tkp in top_k_prompts:
            for v in vocab:
                new_prompts.append(tkp+" "+v)
        current_prompts = new_prompts      
        print("search space size:",len(current_prompts))
    f.close()