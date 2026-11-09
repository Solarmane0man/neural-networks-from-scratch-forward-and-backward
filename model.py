"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
def numerical_gradient(f, x, eps=1e-5):
    # TODO: Estimate the gradient of scalar f w.r.t. array x via central finite differences
    grad = np.zeros_like(x)
    if  x.size == 0:
        return grad
    it = np.nditer(x, flags= ["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old +eps
        fp = f(x)
        x[i] = old - eps
        fm = f(x)
        x[i] = old
        grad[i] = (fp - fm)/(2*eps)
        it.iternext()
    return grad

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    # TODO: Return max relative error between analytic and numeric gradients.
    a = np.asarray(analytic_grad, dtype=float)
    n = np.asarray(numeric_grad, dtype=float)
    if a.size == 0:
        return 0.0
    denom = np.maximum(np.abs(a), np.abs(n))
    denom = np.maximum(denom, tol)          # avoid 0/0 when both entries are 0
    return float(np.max(np.abs(a - n) / denom))

# Step 3 - make_dense
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer.

    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
        Analytic dx/dW/db must match numerical_gradient via gradient_check.
    """
    # TODO: your approach here
    
    W, b = weight_init_fn(in_dim, out_dim)
    params = {'W': np.asarray(W, dtype=float), 'b': np.asarray(b, dtype=float)}

    def forward(x):
        x = np.asarray(x, dtype=float)
        y = x @ params['W'] + params['b']      # bias broadcasts over the batch
        return y, x                            # cache = the input

    def backward(dout, cache):
        x = cache
        dout = np.asarray(dout, dtype=float)
        dx = dout @ params['W'].T              # (N, out) @ (out, in) -> (N, in)
        dW = x.T @ dout                        # (in, N) @ (N, out)  -> (in, out)
        db = dout.sum(axis=0)                  # (N, out) -> (out,)
        return dx, {'W': dW, 'b': db}

    return {'params': params, 'forward': forward, 'backward': backward}

# Step 4 - make_activation
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    if kind == 'relu':
        def fwd(x):
            return np.where(x > 0, x, 0.0), x > 0          # cache = mask of positive entries
        def bwd(dout, mask):
            return dout * mask
    elif kind == 'tanh':
        def fwd(x):
            y = np.tanh(x)
            return y, y
        def bwd(dout, y):
            return dout * (1 - y ** 2)
    elif kind == 'sigmoid':
        def fwd(x):
            y = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return y, y
        def bwd(dout, y):
            return dout * y * (1 - y)
    else:
        raise ValueError(f"unknown activation: {kind}")

    def forward(x):
        return fwd(np.asarray(x, dtype=float))

    def backward(dout, cache):
        return bwd(np.asarray(dout, dtype=float), cache), {}

    return {'params': {}, 'forward': forward, 'backward': backward}

# Step 5 - initialize_weights
def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
         scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    # TODO: your approach here
    if scheme == 'he':
        std = np.sqrt(2.0 / in_dim)
    elif scheme == 'xavier':
        std = np.sqrt(1.0 / in_dim)
    else:
        raise ValueError(f"unknown scheme: {scheme}")
    W = np.random.randn(in_dim, out_dim) * std
    b = np.zeros(out_dim)
    return W, b

# Step 6 - make_loss
def make_loss(kind='cross_entropy'):
    """Return a classification loss_fn(logits, labels) -> (loss, d_logits).

    Inputs to loss_fn:
      logits: (batch, C) float array of raw class scores
      labels: (batch,) int array of class indices in [0, C)
    Outputs:
      loss: Python float, mean scalar loss over the batch (finite)
      d_logits: 
      (batch, C) gradient of loss w.r.t. logits (finite)
    Must pass gradient_check, be minimized by confident correct predictions,
    and stay finite under saturated logits.
    """
    # TODO: your approach here
    if kind != 'cross_entropy':
        raise ValueError(f"unknown loss: {kind}")

    def loss_fn(logits, labels):
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=int)
        n = logits.shape[0]

        z = logits - logits.max(axis=1, keepdims=True)
        log_q = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        loss = float(-log_q[np.arange(n), labels].mean())

        d = np.exp(log_q)
        d[np.arange(n), labels] -= 1.0
        return loss, d / n

    return loss_fn

# Step 7 - make_sequential
def make_sequential(layers):
    """Compose protocol-honoring layers into one sequential model.

    Inputs:
      layers: list of layer dicts, each with
        forward(x) -> (y, cache),
        backward(dout, cache) -> (dx, grads_dict),
        params: dict of ndarrays (possibly empty).

    Returns a dict with:
      forward(x) -> (y, caches)
        y: final activation after applying every layer in order
        caches: opaque structure needed by backward
      backward(dout, caches) -> (dx, grads_list)
        dx: gradient w.r.t. the original input x
        grads_list: list of length len(layers); grads_list[i] is the
          grads_dict from layers[i] ({} for param-free layers)
      params: aggregated live view of all layer params, length len(layers),
        same order as layers (so in-place updates affect the model)
    """
    # TODO: your approach here
    layers = list(layers)
    def forward(x):
      caches = []
      for layer in layers:
        x, c = layer['forward'](x)
        caches.append(c)
      return x,caches
    def backward(dout,caches):
      grads_list = [None] * len(layers)
      for i in reversed(range(len(layers))):
        dout , grads_list[i] = layers[i]['backward'](dout,caches[i])
      return dout, grads_list
    return {'forward': forward,
            'backward': backward,
            'params': [layer['params'] for layer in layers]}

# Step 8 - forward_backward
def forward_backward(model, loss_fn, x, y):
    """Run one full forward-backward sweep on a batch.

    Inputs:
      model: sequential dict with 'forward', 'backward', 'params'
             model['forward'](x) -> (logits, caches)
             model['backward'](d_logits, caches) -> (dx, param_grads)
      loss_fn: callable (logits, y) -> (loss, d_logits)
      x: np.ndarray (batch, in_dim)
      y: np.ndarray (batch,) integer labels

    Returns:
      loss: float, scalar batch loss
      param_grads: nested np.ndarrays matching model['params'] layout
                   (gradients of loss w.r.t. every parameter)
    """
    
    logits, caches = model['forward'](x)
    loss, d_logits = loss_fn(logits,y)
    _, param_grads = model['backward'](d_logits,caches)

    return float(loss), param_grads

# Step 9 - make_optimizer
def make_optimizer(params, lr=1e-2, kind='sgd'):
    """Build an optimizer that updates params in place.

    Inputs:
      params: arrays, possibly nested in lists/dicts (or dict of arrays) to optimize
      lr: float learning rate
      kind: str algorithm name (e.g. 'sgd')

    Returns:
      dict with key 'step'. step(grads) applies one in-place update
      using grads structured like params. Parameter shapes must stay
      unchanged. Repeated steps must reduce a simple convex objective
      within a modest fixed budget and keep values finite.
    """
    import numpy as np
    # TODO: your approach here
    def _flatten(tree):
      if isinstance(tree,np.ndarray):
        return [tree]
      if isinstance(tree,dict):
        return [a for v in tree.values() for a in _flatten(v)]
      if isinstance(tree, (list,tuple)):
        return [a for v in tree for a in _flatten(v)]
      raise TypeError(f"unsupported container: {type(tree)}")
    arrays = _flatten(params)
    if kind == 'momentum':
      vel = [np.zeros_like(a) for a in arrays]
    if kind == 'adam':
        m = [np.zeros_like(a) for a in arrays]  
        v = [np.zeros_like(a) for a in arrays]
        t = [0]
    momentum, beta1, beta2, eps = 0.9, 0.9, 0.999, 1e-8
    def step(grads):
      gs = _flatten(grads)
      if len(gs) != len(arrays):
        raise ValueError("grads structure does not match params")
      if kind == 'sgd':
        for a,g in zip(arrays, gs):
          
          a -= lr*g
      elif kind == 'momentum':
        for a,g ,u in zip(arrays, gs, vel):
          u *= momentum; u+=g
          a -= lr*u
      elif kind == 'adam':
        t[0] +=1
        for a,g,mi,vi in zip(arrays,gs, m, v):
          mi *= beta1; mi += (1 - beta1) * g
          vi *= beta2; vi += (1 - beta2) * g * g
          a -= lr * (mi / (1 - beta1 ** t[0])) / (np.sqrt(vi / (1 - beta2 ** t[0])) + eps)
      else :
        raise ValueError(f"unknown optimizer: {kind}")
    return {'step':step}

# Step 10 - train_step
def train_step(model, loss_fn, optimizer, x_batch, y_batch):
    """Perform one complete optimization step over a minibatch.

    Inputs:
      model: sequential model dict with 'forward', 'backward', and 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x_batch: np.ndarray of shape (B, D)
      y_batch: np.ndarray of shape (B,) integer class labels

    Returns:
      loss: float, scalar batch loss evaluated BEFORE the parameter update.
      Model parameters are updated in place; shapes unchanged and values finite.
    """
    loss, grads = forward_backward(model, loss_fn, x_batch, y_batch)
    optimizer['step'](grads)
    return float(loss)

# Step 11 - train
def train(model, loss_fn, optimizer, x, y, epochs, batch_size, seed=0):
    """Run a deterministic minibatch training loop.

    Inputs:
      model: sequential model dict with 'forward', 'backward', 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x: np.ndarray of shape (N, D) training features
      y: np.ndarray of shape (N,) integer class labels
      epochs: int, number of full passes over the data
      batch_size: int, minibatch size
      seed: int, RNG seed for deterministic shuffling / batching

    Returns:
      history: list[float] of length `epochs`; history[t] is the mean
      train_step loss over minibatches in epoch t.
      Model parameters are updated in place; shapes unchanged.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    history = []
    for _ in range(epochs):
      perm = rng.permutation(n)
      losses = []
      for start in range(0,n,batch_size):
        idx = perm[start:start + batch_size]
        losses.append(train_step(model,loss_fn,optimizer,x[idx],y[idx]))
      history.append(float(np.mean(losses)))
    return history

# Step 12 - design_network
import numpy as np
def design_network(input_dim, num_classes, seed=0):
    """Design and train a net that solves a nonlinear classification task.

    Inputs:
      input_dim: int, feature dimension
      num_classes: int, number of classes
      seed: int, RNG seed for reproducibility

    Returns:
      model: trained sequential model (forward/backward/params)
      metrics: dict with
        'accuracy': float >= 0.90 on an evaluation set,
        'x': np.ndarray (N, input_dim) eval features (N >= 50),
        'y': np.ndarray (N,) integer eval labels.
      The eval set (x, y) must not be linearly separable to high accuracy
      (< 0.82 for a linear classifier), and the model's true accuracy on
      it must match metrics['accuracy'] and be >= 0.90.
    """
    # TODO: your approach here
    def _bands(x,num_classes):
      """Label = which of 2*C equal-width bands along feature 0 (x in [-1, 1]), taken mod C."""

      nb = 2*num_classes
      band = np.clip(((x[:, 0] + 1) / 2 * nb).astype(int), 0, nb - 1)
      return band % num_classes
    rng = np.random.default_rng(seed)
    n_train = 2000*num_classes
    x_tr = rng.uniform(-1, 1, (n_train, input_dim)); y_tr = _bands(x_tr, num_classes)
    x_ev = rng.uniform(-1, 1, (500, input_dim));     y_ev = _bands(x_ev, num_classes)

    width = 32*num_classes
    init =lambda i, o: (rng.standard_normal((i, o)) * np.sqrt(2.0 / i), np.zeros(o))  # He init
    model = make_sequential([
      make_dense(input_dim, width, init), make_activation('relu'),
      make_dense(width, width, init), make_activation('relu'),
      make_dense(width, num_classes, init),]
    )
    loss_fn = make_loss()
    opt = make_optimizer(model['params'], lr=0.005, kind='adam')
    train(model, loss_fn, opt, x_tr, y_tr, epochs=150, batch_size=128, seed=seed)


    acc = float(np.mean(model['forward'](x_ev)[0].argmax(axis=1) == y_ev))
    return model, {'accuracy': acc, 'x': x_ev, 'y': y_ev}

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

