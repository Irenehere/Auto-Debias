# Auto-Debias

## requirments

## related files
    - model
    - data

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
