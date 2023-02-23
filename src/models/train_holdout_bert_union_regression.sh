#!/usr/bin/env bash

for ((i=42; i <= 42; i++))
do
	python train.py \
	 --data_path "../../data/set_holdout_union_regression_v3/" \
	 --model_path "../../models/holdout_bert_union_regression_v3_seed_${i}/" \
	 --model_type "BertForSequenceClassification" \
	 --model_name "bert-base-multilingual-cased" \
	 --tokenizer_type "BertTokenizer" \
	 --tokenizer_name "bert-base-cased" \
	 --column "argumentative_density_union" \
	 --num_labels 1 \
	 --undersample 1000000 \
	 --epochs 8 \
	 --early_stop \
	 --patience 3 \
	 --batch_size 32 \
	 --max_length 512 \
	 --optimizer "AdamW" \
	 --learning_rate 0.00002 \
	 --epsilon 0.000001 \
	 --no_freeze \
	 --seed_value ${i} \
	 --use_cuda
done