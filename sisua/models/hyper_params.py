try:
  from hyperopt import hp, fmin, Trials, STATUS_OK, STATUS_FAIL
  from hyperopt.tpe import suggest as tpe_suggest
  from hyperopt.rand import suggest as rand_suggest
  from hyperopt.pyll import scope
except ImportError as e:
  raise RuntimeError(
      "Cannot import hyperopt for hyper-parameters tuning, error: %s" % str(e))

  def fit_hyper(
      cls,
      inputs: Union[SingleCellOMIC, Iterable[SingleCellOMIC]],
      params: dict = {
          'nlayers': scope.int(hp.choice('nlayers', [1, 2, 3, 4])),
          'hdim': scope.int(hp.choice('hdim', [32, 64, 128, 256, 512])),
          'zdim': scope.int(hp.choice('zdim', [32, 64, 128, 256, 512])),
      },
      loss_name: Text = 'val_loss',
      max_evals: int = 100,
      model_kwargs: dict = {},
      fit_kwargs: dict = {
          'epochs': 64,
          'batch_size': 128
      },
      algorithm: Text = 'bayes',
      seed: int = 8,
      save_path: Text = '/tmp/{model:s}_{data:s}_{loss:s}_{params:s}.hp',
      override: bool = True,
      verbose: bool = False):
    """ Hyper-parameters optimization for given SingleCellModel
    Parameters
    ----------
    model_kwargs : `dict`
      keyword arguments for model construction
    fit_kwargs : `dict`
      keyword arguments for `fit` method

    Example
    -------
    >>> callbacks = [
    >>>     NegativeLogLikelihood(),
    >>>     ImputationError(),
    >>>     CorrelationScores(extras=y_train)
    >>> ]
    >>> i, j = DeepCountAutoencoder.fit_hyper(x_train,
    >>>                                       kwargs={'loss': 'zinb'},
    >>>                                       fit_kwargs={
    >>>                                           'callbacks': callbacks,
    >>>                                           'epochs': 64,
    >>>                                           'batch_size': 128
    >>>                                       },
    >>>                                       loss_name='pearson_mean',
    >>>                                       algorithm='bayes',
    >>>                                       max_evals=100,
    >>>                                       seed=8)
    """
    if isinstance(loss_name, string_types):
      loss_name = [loss_name]
    loss_name = [str(i) for i in loss_name]

    algorithm = str(algorithm.lower())
    assert algorithm in ('rand', 'grid', 'bayes'), \
      "Only support 3 algorithm: rand, grid and bayes; given %s" % algorithm
    # force to turn of keras verbose, it is a big mess to show
    # 2 progress bar at once
    fit_kwargs.update({'verbose': 0})

    # remove unncessary params
    args = inspect.getfullargspec(cls.__init__)
    params = {i: j for i, j in params.items() if i in args.args}

    # processing save_path
    fmt = {}
    for (_, key, spec, _) in string.Formatter().parse(save_path):
      if spec is not None:
        fmt[key] = None
    if isinstance(inputs, (tuple, list)):
      dsname = inputs[0].name if hasattr(inputs[0],
                                         'name') else 'x%d' % len(inputs)
    else:
      dsname = inputs.name if hasattr(inputs, 'name') else 'x'
    kw = {
        'model':
            cls.id,
        'data':
            dsname.replace('_', ''),
        'loss':
            '_'.join(
                [i.replace('val_', '').replace('_', '') for i in loss_name]),
        'params':
            '_'.join(sorted([i.replace('_', '') for i in params.keys()]))
    }
    kw = {i: j for i, j in kw.items() if i in fmt}
    save_path = save_path.format(**kw)
    if os.path.exists(save_path) and not override:
      raise RuntimeError("Cannot override path: %s" % save_path)

    # ====== verbose mode ====== #
    if verbose:
      print(" ======== Tunning: %s ======== " % cls.__name__)
      print("Save path:", save_path)
      print("Model config:", model_kwargs)
      print("Fit config  :", fit_kwargs)
      print("Loss name   :", loss_name)
      print("Algorithm   :", algorithm)
      print("Max evals   :", max_evals)
      print("Search space:")
      for i, j in params.items():
        print("  ", i, j)

    def fit_and_evaluate(*args):
      kw = args[0]
      kw.update(model_kwargs)
      obj = cls(**kw)
      obj.fit(inputs, **fit_kwargs)
      history = obj.history.history
      all_loss = [history[name] for name in loss_name]
      # get min, variance and status, if NaN set to Inf
      loss = 0
      loss_variance = 0
      is_nan = False
      for l in all_loss:
        if np.any(np.isnan(l)):
          is_nan = True
          loss = np.inf
          loss_variance = np.inf
          break
        else:  # first epoch doesn't count
          loss += np.min(l[1:])
          loss_variance += np.var(l[1:])
      loss = loss / len(all_loss)
      loss_variance = loss_variance / len(all_loss)
      return {
          'loss': loss,
          'loss_variance': loss_variance,
          'history': history,
          'status': STATUS_FAIL if is_nan else STATUS_OK,
      }

    def hyperopt_run():
      trials = Trials()
      results = fmin(fit_and_evaluate,
                     space=params,
                     algo=tpe_suggest if algorithm == 'bayes' else rand_suggest,
                     max_evals=int(max_evals),
                     trials=trials,
                     rstate=np.random.RandomState(seed),
                     verbose=verbose)
      history = []
      for t in trials:
        r = t['result']
        history.append({
            'loss': r['loss'],
            'loss_variance': r['loss_variance'],
            'params': {i: j[0] for i, j in t['misc']['vals'].items()},
            'history': r['history'],
            'status': r['status'],
        })
      with open(save_path, 'wb') as f:
        print("Saving hyperopt results to: %s" % save_path)
        dill.dump((results, history), f)

    p = mpi.Process(target=hyperopt_run)
    p.start()
    p.join()

    try:
      with open(save_path, 'rb') as f:
        results, history = dill.load(f)
    except FileNotFoundError:
      results, history = {}, {}

    if verbose:
      print("Best:", results)
    return results, history
