"""Training script

licence: MIT
revision: 2019-06-19 10:38:07
docformat: reStructuredText

"""
import numpy as np
import pandas as pd
import timeit
import pickle
import theano
import theano.tensor as T
import optimizers
import yaml

from theano import shared, function
from scipy import stats
from datetime import datetime as dt

from models import Logit, ResNet, MLP
from core import shared_dataset

# constants
FLOATX = theano.config.floatX

# read data file from .csv
raw_data = pd.read_csv('data-20190702_2.csv')

def main():
    # read configuration file
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    # keep track of time
    config['timestamp'] = dt.now().strftime("%Y-%m-%d %H:%M:%S")

    # defines the inputs and output
    x_data = raw_data.iloc[:, 1:-1]
    y_data = raw_data['mode']-1  # -1 for indexing at 0

    # defines the list of explanatory variable names
    config['variables'] = list(x_data.columns)

    # number of observations
    config['n_obs'] = raw_data.shape[0] 

    # number of variables/choices
    config['n_vars'] = x_data.shape[1]
    config['n_choices'] = len(config['choices'])

    # slicing index for train/valid split
    slice = np.floor(0.7*config['n_obs']).astype(int)
    config['slice'] = slice

    # slices x and y datasets into train/valid
    train_x_data, valid_x_data = x_data.iloc[:slice], x_data.iloc[slice:]
    train_y_data, valid_y_data = y_data.iloc[:slice], y_data.iloc[slice:]

    # load train/valid datasets into shared module
    train_x_shared, train_y_shared = shared_dataset(train_x_data, train_y_data)
    valid_x_shared, valid_y_shared = shared_dataset(valid_x_data, valid_y_data)

    # number of train/valid batches
    n_train_batches = train_y_data.shape[0] // config['batch_size']
    n_valid_batches = valid_y_data.shape[0] // config['batch_size']
    config['n_train_batches'] = n_train_batches
    config['n_valid_batches'] = n_valid_batches

    # Theano tensor variables
    idx = T.lscalar()  # index to [mini]batch
    x = T.matrix('x')
    y = T.ivector('y')

    if config['model_type'] == 'ResNet':
        # create ResNet model
        model = ResNet(
            input=x, choice=y, 
            n_vars=config['n_vars'], 
            n_choices=config['n_choices'], 
            n_layers=config['n_layers']
        )
        cost = model.negative_log_likelihood(y)
        opt = optimizers.RMSProp(model.params)
        updates = opt.run_update(
            cost, model.params, 
            masks=model.params_mask, learning_rate=config['learning_rate']
        )
    elif config['model_type'] == 'MLP':
        # create MLP model
        model = MLP(
            input=x, choice=y, 
            n_vars=config['n_vars'], 
            n_choices=config['n_choices'], 
            n_layers=config['n_layers']
        )
        cost = model.negative_log_likelihood(y)
        opt = optimizers.RMSProp(model.params)
        updates = opt.run_update(
            cost, model.params, 
            learning_rate=config['learning_rate']
        )
    elif config['model_type'] == 'MNL':
        # create MNL model
        config['n_layers'] = 0
        model = Logit(
            input=x, choice=y, 
            n_vars=config['n_vars'], 
            n_choices=config['n_choices']
        )
        cost = model.negative_log_likelihood(y)
        opt = optimizers.RMSProp(model.params)
        updates = opt.run_update(
            cost, model.params, 
            masks=model.params_mask, learning_rate=config['learning_rate']
        )
    print(config['model_type'])

    batch_size = config['batch_size']

    model.train_model = function(
        inputs=[idx],
        outputs=cost,
        updates=updates,
        allow_input_downcast=True,
        givens={
            x: train_x_shared[idx * batch_size: (idx + 1) * batch_size],
            y: train_y_shared[idx * batch_size: (idx + 1) * batch_size],
        },
    )

    model.validate_model = function(
        inputs=[],
        outputs=cost,
        allow_input_downcast=True,
        givens={
            x: valid_x_shared,
            y: valid_y_shared,
        },
    )

    model.predict_model = function(
        inputs=[],
        outputs=model.errors(y),
        allow_input_downcast=True,
        givens={
            x: valid_x_shared,
            y: valid_y_shared,
        },
    )

    model.hessians = function(
        inputs=[idx],
        outputs=model.get_gessians(y),
        allow_input_downcast=True,
        givens={
            x: train_x_shared[idx * batch_size: (idx + 1) * batch_size],
            y: train_y_shared[idx * batch_size: (idx + 1) * batch_size],
        },
    )

    valid_freq = min(200, n_train_batches)
    best_validation_ll = np.inf
    start_time = timeit.default_timer()
    done_looping = False
    epoch = 0
    step = 0

    filename = '{}{}_bestmodel.pkl'.format(config['model_type'], config['n_layers'])
    training_frame = pd.DataFrame(
        columns=['epoch', 'minibatch', 'batches', 'train_ll', 'valid_ll', 'valid_err']
    )

    while (epoch < config['n_epochs']) and (not done_looping):
        epoch = epoch + 1
        training_ll = 0
        for i in range(n_train_batches):
            # accumulating average score
            minibatch_ll = model.train_model(i)
            training_ll = (training_ll * i + minibatch_ll)/(i + 1)
            
            iteration = (epoch - 1) * n_train_batches + i

            if (iteration + 1) % valid_freq == 0:
                validation_ll = np.sum(model.validate_model())
                #################################
                # track and save training stats #
                #################################
                training_step = {
                    'epoch': epoch, 
                    'minibatch': i + 1, 
                    'batches': n_train_batches, 
                    'train_ll': training_ll * n_train_batches, 
                    'valid_ll': validation_ll, 
                    'valid_err': None,
                }
                training_frame.loc[step] = training_step
                #################################
                
                if validation_ll < best_validation_ll:
                    print(('epoch {:d}, minibatch {:d}/{:d}, '
                        'validation likelihood {:.2f}').format(
                            epoch, i + 1, n_train_batches, validation_ll))

                    # improve patience if loss improvement is good enough
                    if validation_ll < best_validation_ll * config['improvement_threshold']:
                        config['patience'] = max(config['patience'], iteration * config['patience_increase'])

                    # keep track of best validation score
                    best_validation_ll = validation_ll

                    # check prediction accuracy
                    error = np.mean(model.predict_model())

                    # set valid error in training frame
                    training_frame.loc[step, 'valid_err'] = error
                    print('validation error  {:.2%}'.format(error))

                    # save the best model
                    with open(filename, 'wb') as f:
                        pickle.dump([model, config], f)

                step = step + 1

            if epoch > 200:
                done_looping = True
                break

    end_time = timeit.default_timer()
    run_time = end_time - start_time

    #############################
    # parameter extraction step #
    #############################
    print('processing model statistics...')
    with open(filename, 'rb') as f:
        model, config = pickle.load(f)

    # format model beta params and ASCs
    model_stat = {}
    if config['model_type'] in ['MNL', 'ResNet']:
        beta = model.beta.eval().round(3)
        beta_df = pd.DataFrame(beta, index=config['variables'], columns=config['choices'])
        model_stat['beta_params'] = beta_df

        asc = model.params[1].eval().round(3)
        asc_df = pd.DataFrame(asc.reshape((1,-1)), index=['ASC'], columns=config['choices'])
        model_stat['asc_params'] = asc_df

        # compute std. err and t-stat
        h = np.mean([model.hessians(i) for i in range(n_train_batches)], axis=0)
        model_stat['beta_stderr'] = pd.DataFrame(
            np.sqrt(1/np.diag(h[0]).reshape((config['n_vars'], config['n_choices'])))/(batch_size-1), 
            index=config['variables'], 
            columns=config['choices']
        )
        model_stat['beta_t_stat'] = pd.DataFrame(
            beta / model_stat['beta_stderr'], index=config['variables'], columns=config['choices'])
        
    # format ResNet residual matrix
    if config['model_type'] == 'ResNet':
        model_stat['residual_matrix'] = []
        for l in range(config['n_layers']):
            # create a pandas correlation matrix table
            mat = model.resnet_layers[l].params[0].eval().round(2)
            df = pd.DataFrame(
                data=mat, index=config['choices'], columns=config['choices'])
            model_stat['residual_matrix'].append(df)

    # misc: runtime and training curves
    model_stat['run_time'] = str(np.round(run_time / 60., 3))+' minutes'
    model_stat['training_frame'] = training_frame

    # re-save model, configuration and statistics     
    with open(filename, 'wb') as f:
        pickle.dump([model, config, model_stat], f)        
    #############################

    # print final verbose output
    print(('Optimization complete with best validation likelihood of {:.2f}, '
        'and validation error of {:.2%}').format(best_validation_ll, error))
    print(('The code run for {:d} epochs, with {:.2f} epochs/sec').format(
            epoch, 1. * epoch / run_time))
    print('training complete')

if __name__ == "__main__":
    main()
