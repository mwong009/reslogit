#%%
import os
import numpy as np
import pandas as pd
import timeit
import pickle
import theano
import theano.tensor as T
import optimizers
import matplotlib.pyplot as plt

from theano import shared, function
from scipy import stats
from models import Logit, ResNet
from core import shared_dataset

FLOATX = theano.config.floatX

batch_size = 20
learning_rate = 1e-3
layers = 1
n_epochs = 1000

data = pd.read_csv('data-20190617.csv')
# data['trip_length_km'] = stats.boxcox(data['trip_length_km']+1e-7, 0.191)
# data['trip_duration_min'] = stats.boxcox(data['trip_duration_min']+1e-7, 0.20529)
x_var = data.loc[:, 'weekend':'Non disponible']
y_var = data.loc[:, 'activity_choice']

n = data.shape[0]
m = x_var.shape[1]
slice = np.floor(0.7*n).astype(int)

train_x_var, valid_x_var = x_var.iloc[:slice], x_var.iloc[slice:]
train_y_var, valid_y_var = y_var.iloc[:slice], y_var.iloc[slice:]

train_x_shared, train_y_shared = shared_dataset(train_x_var, train_y_var)
valid_x_shared, valid_y_shared = shared_dataset(valid_x_var, valid_y_var)

n_train_batches = train_y_var.shape[0] // batch_size
n_valid_batches = valid_y_var.shape[0] // batch_size

index = T.lscalar()  # index to [mini]batch

x = T.matrix('x')
y = T.ivector('y')

# model = Logit(input=x, choice=y, n_vars=m, n_choices=10)
# cost = model.negative_log_likelihood(y)
# opt = optimizers.RMSProp(model.params)
# updates = opt.run_update(cost, model.params)

model = ResNet(input=x, choice=y, n_vars=m, n_choices=10, n_layers=8)
cost = model.negative_log_likelihood(y)
opt = optimizers.RMSProp(model.resnet_params)
updates = opt.run_update(cost, model.resnet_params)

train_model = function(
    inputs=[index],
    outputs=cost,
    updates=updates,
    allow_input_downcast=True,
    givens={
        x: train_x_shared[index * batch_size: (index + 1) * batch_size],
        y: train_y_shared[index * batch_size: (index + 1) * batch_size],
    },
)

validate_model = function(
    inputs=[],
    outputs=cost,
    allow_input_downcast=True,
    givens={
        x: valid_x_shared,
        y: valid_y_shared,
    },
)

predict_model = function(
    inputs=[index],
    outputs=model.errors(y),
    allow_input_downcast=True,
    givens={
        x: valid_x_shared[index * batch_size: (index + 1) * batch_size],
        y: valid_y_shared[index * batch_size: (index + 1) * batch_size],
    },
)

hessians = function(
    inputs=[index],
    outputs=model.get_gessians(y),
    allow_input_downcast=True,
    givens={
        x: train_x_shared[index * batch_size: (index + 1) * batch_size],
        y: train_y_shared[index * batch_size: (index + 1) * batch_size],
    },
)

patience = 5000  # look as this many examples regardless
patience_increase = 2
improvement_threshold = 0.9999
validation_frequency = n_train_batches

best_validation_likelihood = np.inf
start_time = timeit.default_timer()

done_looping = False
epoch = 0

while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in range(n_train_batches):
        # n = np.random.randint(0, n_train_batches)
        minibatch_avg_cost = train_model(minibatch_index)
        iter = (epoch - 1) * n_train_batches + minibatch_index

        if (iter + 1) % validation_frequency == 0:
            validation_likelihood = validate_model()
            likelihood = np.sum(validation_likelihood)

            print(('epoch {:d}, minibatch {:d}/{:d}, '
                    'validation likelihood {:.2f}').format(
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    likelihood,
                )
            )

            if likelihood < best_validation_likelihood:
                # improve patience if loss improvement is good enough
                if likelihood < best_validation_likelihood * improvement_threshold:
                    patience = max(patience, iter * patience_increase)

                best_validation_likelihood = likelihood

                validation_predict = [predict_model(i)
                                        for i in range(n_valid_batches)]
                error = np.mean(validation_predict)
                print(('                             '
                        'validation error     {:.2%}').format(error))

                # save the best model
                with open('best_model.pkl', 'wb') as f:
                    pickle.dump(model, f)

        if (patience <= iter) and (epoch > 10):
            done_looping = True
            break

end_time = timeit.default_timer()
print(('Optimization complete with best validation likelihood of {:.2f}, '
        'and validation error of {:.2%}').format(best_validation_likelihood, error))
print(('The code run for {:d} epochs, with {:.2f} epochs/sec').format(
        epoch,
        1. * epoch / (end_time - start_time),
    )
)

print('statistics...')
model_hessians = np.mean([hessians(i) for i in range(n_train_batches)], axis=0)
model_stderr = [np.sqrt(1/h) for h in model_hessians]
model_tstat = [b.get_value()/s for b, s in zip(model.params, model_stderr)]
# print('hessians', pd.DataFrame(model_hessians[0]))

#%%
print('statistics...')
np.set_printoptions(precision=4, suppress=True)
model_hessians = np.sum([hessians(i) for i in range(n_train_batches)], axis=0)
model_stderr = [np.sqrt(1/h) for h in model_hessians]
model_tstat = [b.get_value()/s for b, s in zip(model.params, model_stderr)]

#%%
df = pd.DataFrame(model.params[0].eval())
df.to_csv('beta_init.csv', index=False, header=False)
print(df.shape)
df = pd.DataFrame(model.params[1].eval())
df.to_csv('beta_0_init.csv', index=False, header=False)
print(df.shape)

#%%
def hinton(matrix, tstat=None, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max(axis=1)) / np.log(2))
    matrix = (matrix - np.mean(matrix, axis=1, keepdims=True))/ np.std(matrix, axis=1, keepdims=True)

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):

        size = np.sqrt(np.abs(w)/max_weight[y])
        color = 'white' if w > 0 else 'black'

        if tstat is not None:
            if tstat[x, y] >= 1:
                edge = 'red'
            else:
                edge = color
        else:
            edge = color

        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=edge)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


#%%
tstat = np.abs(model_tstat[0])
p = model.params[0].get_value()
hinton(p, tstat=(tstat <= 1.96))


#%%
