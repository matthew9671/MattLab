import jax
from jax import jit
import jax.random as jr
from functools import partial
from tqdm import trange

class Trainer:
    """
    model: a pytree node
    loss (key, params, model, data, **train_params) -> (loss, aux)
        Returns a loss (a single float) and an auxillary output (e.g. posterior)
    init (key, model, data, **train_params) -> (params, opts)
        Returns the initial parameters and optimizers to go with those parameters
    update (params, grads, opts, model, aux, **train_params) -> (params, opts)
        Returns updated parameters, optimizers
    """
    def __init__(self, model, 
                 train_params=None, 
                 init=None, 
                 loss=None, 
                 update=None):
        # Trainer state
        self.params = None
        self.model = model

        if train_params is None:
            train_params = dict()

        self.train_params = train_params

        if init is not None:
            self.init = init
        if loss is not None:
            self.loss = loss
        if update is not None: 
            self.update = update

    @partial(jit, static_argnums=(0,))
    def train_step(self, key, params, data, opt_states, **train_params):
        model = self.model
        results = \
            jax.value_and_grad(
                lambda params: partial(self.loss, **train_params)(key, model, data, params), has_aux=True)(params)
        (loss, aux), grads = results
        params, opts = self.update(params, grads, self.opts, opt_states, model, aux, **train_params)
        return params, opts, (loss, aux)

    def test_step(self, key, params, model, data):
        loss_out = self.loss(key, params, model, data)
        return loss_out

    """
    Callback: a function that takes training iterations and relevant parameter
        And logs to WandB
    """
    def train(self, data_dict, max_iters, callback=None, key=None):

        if callback is None:
            callback = lambda trainer, loss_out: None
        if key is None:
            key = jr.PRNGKey(0)

        model = self.model
        train_data = data_dict["train_data"]

        init_key, key = jr.split(key, 2)

        # Initialize optimizer
        self.params, self.opts, opt_states = self.init(init_key, model, train_data, **self.train_params)
        self.train_losses = []
        self.test_losses = []
        self.val_losses = []

        pbar = trange(max_iters)
        pbar.set_description("[jit compling...]")
        for itr in pbar:
            this_key, key = jr.split(key, 2)
            self.params, opt_states, loss_out = \
                self.train_step(this_key, self.params, train_data, opt_states, **self.train_params)
            callback(self, loss_out)
            loss, aux = loss_out
            self.train_losses.append(loss)
            pbar.set_description("LP: {:.3f}".format(loss))