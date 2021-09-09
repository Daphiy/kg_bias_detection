import torch
import argparse
import os
import pandas as pd
from BiasEvaluator import *
from utils import *
from predict_tails import *
from Wiki5m import Wiki5m
from model_wrapper import *


##### Global Parameters #####
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Parser
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--measure', type=str, 
                         help="Whether to run translational or PDP & DPD", 
                         default='TLB')        
    parser.add_argument('--dataset', type=str, 
                         help="name of the dataset - fb15k237 or wikidata5m", 
                         default='wikidata5m') 
    parser.add_argument('--bs', type=int, default=256, help="Classifier batch size")
    args = parser.parse_args()
    
    dataset = Wiki5m()
    # Trained Embedding Model Path
    MODEL_PATH = os.path.join('./trained_models/', dataset.name, args.embedding+".pkl")
    if args.embedding_path:
        MODEL_PATH = args.embedding_path # override default if specifying a full path
    print("Load embedding model from: {}".format(MODEL_PATH))
    
    # Init dataset and relations of interest
    target_relation, bias_relations = suggest_relations(dataset.name)

    # Load trained MLP predictions
    if args.predictions_path:
        PREDS_PATH = args.predictions_path
        PREDS_DIR = os.path.split(PREDS_PATH)[0]
        if not os.path.exists(PREDS_DIR):
            os.makedirs(PREDS_DIR)
    else:
        PREDS_PATH = None


    # Init embed model and classifier parameter
    model_args = {'embedding_model_path':MODEL_PATH}
    classifier_args = {'epochs':args.epochs, 
                       'type':'rf', 
                       'num_classes':11,
                       'batch_size': args.bs, 
                       'max_depth':4
                       }
    
    # Init Evaluator
    if args.measure == "TLB" or args.measure == 'translational': 
        measures = [TranslationalLikelihood()]
    else: 
        measures = [DemographicParity(),PredictiveParity()]
    evaluator = BiasEvaluator(dataset, measures)
    evaluator.set_target_relation(target_relation)
    evaluator.set_bias_relations(bias_relations)
    model = torch.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
    evaluator.set_trained_model(model)
    
    # Evaluate Bias
    result, preds_df = eval_bias(evaluator=evaluator,
                       classifier_args=classifier_args,
                       model_args=model_args,
                       bias_relations=bias_relations,
                       bias_measures=measures,
                       preds_df_path=PREDS_PATH
                    )
    print(result)
    # Save Detection result
    save_result(result, dataset, args)
    preds_df.to_csv()
