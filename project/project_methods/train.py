''' This file contains the code for training and evaluating a model.
'''

from model import LogisticRegression, MajorityBaseline, Model, SupportVectorMachine, MODEL_OPTIONS


def init_model(args: object, num_features: int) -> Model:
    '''
    Initialize the appropriate model from command-line arguments.

    Args:
        args (object): the argparse Namespace mapping arguments to their values.
        num_features (int): the number of features (i.e. dimensions) the model will have

    Returns:
        Model: a Model object initialized with the hyperparameters in args.
    '''

    if args.model == 'majority_baseline':
        model = MajorityBaseline()
    
    elif args.model == 'svm':
        model = SupportVectorMachine(
            num_features=num_features, 
            lr0=args.lr0, 
            C=args.reg_tradeoff)

    elif args.model == 'logistic_regression':
        model = LogisticRegression(
            num_features=num_features, 
            lr0=args.lr0, 
            sigma2=args.reg_tradeoff)
    
    return model

