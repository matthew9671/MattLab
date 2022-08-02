import jax
import jax.random as jr

class Trainer:
    """
    model: a pytree node
    loss (key, params, model, data)
        Returns a loss (a single float) and an auxillary output (e.g. posterior)
    init (key, model, data)
        Returns (params, opts)
    update (params, grads, opts, model, aux)
        Returns (params, opts, model)
    """
    def __init__(self, model, init, loss, update):
        # Trainer state
        self.params = None
        self.model = model
        self.init = init
        self.loss = loss
        self.update = update

    def train_step(self, key, params, model, data, opts):
        results = \
            jax.value_and_grad(lambda params: self.loss(key, params, model, data), has_aux=True)(params)
        (loss, aux), grads = results
        params, opts, model = self.update(params, grads, opts, model, aux)
        return params, model, opts, (loss, aux)

    def test_step(self, key, params, model, data):
        loss_out = self.loss(key, params, model, data)
        return loss_out

    """
    Callback: a function that takes training iterations and relevant parameter
        And logs to WandB
    """
    def train(self, data_dict, max_iters, callback=None, key=None):

        if callback is None:
            callback = lambda x: None
        if key is None:
            key = jr.PRNGKey(0)

        model = self.model
        train_data = data_dict["train_data"]

        init_key, key = jr.split(key, 2)

        # Initialize optimizer
        self.params, opts = self.init(init_key, model, train_data)
        self.train_losses = []
        self.test_losses = []
        self.val_losses = []

        for itr in range(max_iters):
            this_key, key = jr.split(key, 2)
            self.params, model, opts, loss_out = \
                self.train_step(this_key, self.params, model, train_data, opts)
            callback(loss_out)
            loss, aux = loss_out
            self.train_losses.append(loss)