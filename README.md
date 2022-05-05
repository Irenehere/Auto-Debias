# Auto-Debias

## requirments
    - numpy
    - torch
    - transformers 

## related files
We release the debiased model [here](https://drive.google.com/drive/folders/1MjmUXxfoGhOVGxpRSwsU9Pt1EYF4uaIL?usp=sharing).  

## generated biased prompt via beam search

```
python generate_prompts.py \
--debias_type   gender or race 
--model_type   bert or roberta or albert
--model_name_or_path  bert-base-uncased, etc
```

## debiasing language models 
```
python auto-debias.py
--debias_type    gender or race 
--model_type   bert or roberta or albert
--model_name_or_path  bert-base-uncased, etc
--prompts_file prompts_bert-base-uncased_gender
```

## evaluation
 ### SEAT
 We run the [SEAT](https://github.com/pliang279/sent_debias) using the code from Liang et al.
 
 ### GLUE
 We run [GLUE](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) from transformers.
