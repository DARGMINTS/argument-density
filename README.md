# Predicting Argument Density from Text
===============

Sample source code and models for our [NLDB 2022](http://www.wikicfp.com/cfp/servlet/event.showcfp?eventid=152113&copyownerid=320) paper: [Predicting argument density from multiple annotations](https://link.springer.com/chapter/10.1007/978-3-031-08473-7_21)

> **Abstract:** Annotating a corpus with argument structures is a complex task, and it is even more challenging when addressing text genres where argumentative discourse markers do not abound. We explore a corpus of opinion articles annotated by multiple annotators, providing diverse perspectives of the argumentative content therein. New annotation aggregation methods are explored, diverging from the traditional ones that try to minimize presumed errors from annotator disagreement. The impact of our methods is assessed for the task of argument density prediction, seen as an initial step in the argument mining pipeline. We evaluate and compare models trained for this regression task in different generated datasets, considering their prediction error and also from a ranking perspective. Results confirm the expectation that addressing argument density from a ranking perspective is more promising than looking at the problem as a mere regression task. We also show that probabilistic aggregation, which weighs tokens by considering all annotators, is a more interesting approach, achieving encouraging results as it accommodates different annotator perspectives. The code and models are publicly available at https://github.com/DARGMINTS/argument-density.

**Authors:** Gil Rocha, Bernardo Leite, Luís Trigo, Henrique Lopes Cardoso, Rui Sousa-Silva, Paula Carvalho, Bruno Martins, Miguel Won

If you use this research in your work, please kindly cite us:
```bibtex
@inproceedings{rocha_2022_argdens,
	title        = {Predicting Argument Density from Multiple Annotations},
	author       = {Rocha, Gil and Leite, Bernardo and Trigo, Lu{\'i}s and Cardoso, Henrique Lopes and Sousa-Silva, Rui and Carvalho, Paula and Martins, Bruno and Won, Miguel},
	year         = 2022,
	booktitle    = {Natural Language Processing and Information Systems},
	publisher    = {Springer International Publishing},
	address      = {Cham},
	pages        = {227--239},
	isbn         = {978-3-031-08473-7},
	editor       = {Rosso, Paolo and Basile, Valerio and Mart{\'i}nez, Raquel and M{\'e}tais, Elisabeth and Meziane, Farid}
}
```

## Main Features
* Training and inference scripts for argument density prediction
* Fine-tuned models for Portuguese argument density prediction

## Prerequisites
Python 3 (tested with version 3.8.5 on Windows 10)

## Installation and Configuration
1. Clone this project:
    ```python
    git clone https://github.com/DARGMINTS/argument-density
    ```
2. Install the Python packages from [requirements.txt](https://github.com/DARGMINTS/argument-density/blob/main/requirements.txt). If you are using a virtual environment for Python package management, you can install all python packages needed by using the following bash command:
    ```bash
    cd argument-density/
    pip install -r requirements.txt
    ```

## Usage
You can use this code for **training** your own argument density prediction models and **inference/prediction**.

**Note about data**: In `data/set_holdout_union_regression_v3`, you find examples of data for train, validation, and test sets. Notice that `###.paragraphs.json` contains *random* written paragraphs and `###.labels.json` contains *random* density values. Please contact the authors if you intend to use the article corpus, which consists of 373 Portuguese opinion articles.

### Training 
1. Go to `src/models`. The file `train.py` is responsible for the training routine. Type the following command to read the description of the parameters:

	```bash
	python train.py -h
	```

	You can also run the example training script (linux and mac) `train_holdout_bert_union_regression.sh`:
	```bash
	bash train_holdout_bert_union_regression.sh
	```

	The previous script will start the training routine with predefined parameters:
	```python
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
	```

2. In the end, all model information is available at `models/model_path`. The information includes the best model checkpoint (`*.bin` file), training report (`*.report.txt`), predictions and true_labels in `###.json` format.

### Inference/Prediction
Go to `src/models`. The file `run_predictions.py` is responsible for the inference routine given a certain *PRE_TRAINED_MODEL_NAME*, *MODEL_FINE_TUNED* and *SAMPLE_TEXT*.

Example/Demo:

1.  Change *PRE_TRAINED_MODEL_NAME*, *MODEL_FINE_TUNED* and *SAMPLE_TEXT* variables in `run_predictions.py`:
    ```python
	PRE_TRAINED_MODEL_NAME = 'bert-base-multilingual-cased'
    MODEL_FINE_TUNED = '../../models/holdout_bert_union_regression_v3_seed_42/model_best_epoch.bin'
    SAMPLE_TEXT = 'Este é um exemplo em Português.'
    ```
2.  Run `run_predictions.py`
3.  See output (it should be a number between 0 and 1)

### Checkpoints
We provide [three fine-tuned models](https://uporto-my.sharepoint.com/:f:/g/personal/up201404464_up_pt/EhnznzaNWQJAtUA7uRA4WjQBGX5_L3OiwkiYKujrR8a9aQ?e=e0BO8X) for Portuguese argument density prediction (see corresponding results from our article -- Table 3 -- first column).

Recall that we carry out 10-fold cross-validation. In total, ten iterations have been performed per dataset. Each of the above models results from one iteration (density prediction results are similar between iterations). Please, contact the authors if you want to experiment with additional models.

## Issues and Usage Q&A
To ask questions, report issues or request features, please use the GitHub Issue Tracker.

## Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks in advance!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
### Project
This project is released under the **MIT** license. For details, please see the file [LICENSE](https://github.com/DARGMINTS/argument-density/blob/main/LICENSE) in the root directory.

### Commercial Purposes
A commercial license may also be available for use in industrial projects, collaborations or distributors of proprietary software that do not wish to use an open-source license. Please contact the author if you are interested.

## Acknowledgements
This research has been supported by DARGMINTS (POCI/01/0145/FEDER/031460), LIACC (FCT/UID/CEC/0027/2020) and CLUP (UIDB/00022/2020), funded by Fundação para a Ciência e a Tecnologia (FCT).

## Contact Persons
* Gil Rocha, gil.rocha@fe.up.pt
* Bernardo Leite, bernardo.leite@fe.up.pt
* Luís Trigo, ltrigo@fe.up.pt
* Henrique Lopes Cardoso, hlc@fe.up.pt