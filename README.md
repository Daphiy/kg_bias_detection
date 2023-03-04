# Towards Automatic Bias Detection in Knowledge Graphs
Repository for short paper accepted in EMNLP 2021 Findings, you may find our paper [here](
https://aclanthology.org/2021.findings-emnlp.321/).

## Citation

@inproceedings{keidar-etal-2021-towards-automatic,
    title = "Towards Automatic Bias Detection in Knowledge Graphs",
    author = "Keidar, Daphna  and
      Zhong, Mian  and
      Zhang, Ce  and
      Shrestha, Yash Raj  and
      Paudel, Bibek",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.321",
    doi = "10.18653/v1/2021.findings-emnlp.321",
    pages = "3804--3811",
    abstract = "With the recent surge in social applications relying on knowledge graphs, the need for techniques to ensure fairness in KG based methods is becoming increasingly evident. Previous works have demonstrated that KGs are prone to various social biases, and have proposed multiple methods for debiasing them. However, in such studies, the focus has been on debiasing techniques, while the relations to be debiased are specified manually by the user. As manual specification is itself susceptible to human cognitive bias, there is a need for a system capable of quantifying and exposing biases, that can support more informed decisions on what to debias. To address this gap in the literature, we describe a framework for identifying biases present in knowledge graph embeddings, based on numerical bias metrics. We illustrate the framework with three different bias measures on the task of profession prediction, and it can be flexibly extended to further bias definitions and applications. The relations flagged as biased can then be handed to decision makers for judgement upon subsequent debiasing.",
}

## Data
### FB15K-237
You may download our trained models from [here](https://polybox.ethz.ch/index.php/s/pLp8Bmp9abrytIQ) in directory `trained_models/`, and uncompress it.

### Wikidata 5M
Get Wikidata5m Pre-trained embeddings (TransE, DistMult, ComplEx, RotatE) from [here](https://graphvite.io/docs/latest/pretrained_model.html), and put inside the directory `data/wiki5m`. Since we only work around human-related triples, we filtered and saved needed entities and relations as `human_ent_rel_sorted_list.pkl` in directory `data/wiki5m`. 

Run the following commands to first save human-relate embeddings, and then wrap into its corresponding pykeen trained model which will be saved in the directory `trained_models/wiki5m`
```
python process_wiki5m.py
mkdir -p trained_models/wiki5m
python wrap_wiki5m.py
```
## Classification 
To classify the entities according to the target relation, please refer to the code in experiments/run_tail_prediction.py
In the paper as well as the code files, the target relation is profession - meaning that we train a classifier on the task of predicting the profession for each entity. 

Pre-computed dataframes with the tail predictions (i.e. classifications) for profession in each of the embedding methods can be found under the folder preds_dfs. These can be used to directly calculate the bias measurements. 
