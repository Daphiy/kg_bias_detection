import torch
import argparse
import os
import pandas as pd
from pykeen.datasets import get_dataset, YAGO310, FB15k237
from BiasEvaluator import *
from utils import *
from predict_tails import *

##### Global Parameters #####
datasets = {'fb15k237': FB15k237()}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    LOCAL_PATH_TO_EMBEDDING = '/Users/alacrity/Documents/uni/Fairness/'

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fb15k237',
                        help="Dataset name, must be one of fb15k237. Default to fb15k237.")
    parser.add_argument('--embedding', type=str, default='trans_e',
                        help="Embedding name, must be one of complex, conv_e, distmult, rotate, trans_d, trans_e. \
                              Default to trans_e")
    parser.add_argument('--embedding_path', type=str, 
                        help="Specify a full path to your trained embedding model. It will override default path \
                            inferred by dataset and embedding")
    parser.add_argument('--predictions_path', type=str, 
                        help='path to predictions used in parity distance, specifying \
                             it will override internal inferred path')
    parser.add_argument('--epochs', type=int, 
                         help="Number of training epochs of link prediction classifier (used for DP & PP), default to 100", 
                         default=100)
    args = parser.parse_args()
    
    # Trained Embedding Model Path
    embedding_model_path_suffix = "replicates/replicate-00000/trained_model.pkl"
    MODEL_PATH = os.path.join(LOCAL_PATH_TO_EMBEDDING, args.dataset, args.embedding, embedding_model_path_suffix)
    if args.embedding_path:
        MODEL_PATH = args.embedding_path # override default if specifying a full path
    print("Load embedding model from: {}".format(MODEL_PATH))
    
    # Init dataset and relations of interest
    dataset = datasets[args.dataset]
    target_relation, bias_relations = suggest_relations(args.dataset)

    # Load trained MLP predictions
    if args.dataset == 'fb15k237':
        sfx = target_relation.split("/")[-1]
    else:
        sfx = target_relation
    if args.predictions_path:
        PREDS_PATH = args.predictions_path
        PREDS_DIR = os.path.split(PREDS_PATH)[0]
        if not os.path.exists(PREDS_DIR):
            os.makedirs(PREDS_DIR)
    else:
        PREDS_PATH = None
    print(f"Load MLP classifier from : {PREDS_PATH}")

    # Init embed model and classifier parameter
    model_args = {'embedding_model_path':MODEL_PATH}
    classifier_args = {'epochs':args.epochs, "type":'mlp', 'num_classes':6}
    
    # Init Evaluator
    measures = [TranslationalLikelihood()] #[DemographicParity(), PredictiveParity(), 
    evaluator = BiasEvaluator(dataset, measures)
    evaluator.set_target_relation(target_relation)
    evaluator.set_bias_relations(bias_relations)
    evaluator.set_trained_model(torch.load(MODEL_PATH, map_location=DEVICE))
    
    # Evaluate Bias
    result = eval_bias(evaluator=evaluator,
                       classifier_args=classifier_args,
                       model_args=model_args,
                       bias_relations=bias_relations,
                       bias_measures=measures,
                       preds_df_path=PREDS_PATH
                    )
    print(result)
    # Save Detection result
    save_result(result, dataset, args)
