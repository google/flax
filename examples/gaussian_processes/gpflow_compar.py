def gpflow_fit(data):
    import gpflow
    import tensorflow as tf
    from gpflow.ci_utils import ci_niter

    X, Y = data

    M = 5
    kernel = gpflow.kernels.SquaredExponential()
    Z = X[:M, :].copy()  # Initialize inducing locations to the first M inputs in the dataset

    m = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), Z, num_data=15)

    @tf.function
    def optimization_step(optimizer, model: gpflow.models.SVGP, batch):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(model.trainable_variables)
            objective = - model.elbo(batch)
            grads = tape.gradient(objective, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return objective

    def run_adam(model, iterations):
        """
        Utility function running the Adam optimizer

        :param model: GPflow model
        :param interations: number of iterations
        """
        # Create an Adam Optimizer action
        logf = []
        #train_it = iter(train_dataset.batch(minibatch_size))
        adam = tf.optimizers.Adam()
        for step in range(iterations):
            elbo = - optimization_step(adam, model, data)
            if step % 10 == 0:
                logf.append(elbo.numpy())
        return logf

    maxiter = ci_niter(FLAGS.num_epochs)
    logf = run_adam(m, maxiter)

    gpflow.utilities.print_summary(m)

    return m