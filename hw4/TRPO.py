import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
import copy
from utils import *
import scipy as sp


def normc_initializer(std=1.0):
    """
	Initialize array with normalized columns
	"""

    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


def dense(x, size, name, weight_init=None):
    """
	Dense (fully connected) layer
	"""
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())

    return tf.matmul(x, w) + b


def fancy_slice_2d(X, inds0, inds1):
    """
	Like numpy's X[inds0, inds1]
	"""
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(X), tf.int64)
    ncols = shape[1]
    Xflat = tf.reshape(X, [-1])
    return tf.gather(Xflat, inds0 * ncols + inds1)




def discount(x, gamma):
    """
	Compute discounted sum of future values
	out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
	"""
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def explained_variance_1d(ypred, y):
    """
	Var[ypred - y] / var[y].
	https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
	"""
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def categorical_sample_logits(logits):
    """
	Samples (symbolically) from categorical distribution, where logits is a NxK
	matrix specifying N categorical distributions with K categories

	specifically, exp(logits) / sum( exp(logits), axis=1 ) is the
	probabilities of the different classes

	Cleverly uses gumbell trick, based on
	https://github.com/tensorflow/tensorflow/issues/456
	"""
    U = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(U)), dimension=1)


def pathlength(path):
    return len(path["reward"])


class LinearValueFunction(object):
    coef = None

    def fit(self, X, y):
        Xp = self.preproc(X)
        A = Xp.T.dot(Xp)
        nfeats = Xp.shape[1]
        A[np.arange(nfeats), np.arange(nfeats)] += 1e-3  # a little ridge regression
        b = Xp.T.dot(y)
        self.coef = np.linalg.solve(A, b)

    def predict(self, X):
        if self.coef is None:
            return np.zeros(X.shape[0])
        else:
            return self.preproc(X).dot(self.coef)

    def preproc(self, X):
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X) / 2.0], axis=1)


class NnValueFunction(object):
    #	'''
    def __init__(self, ob_dim=10, params=[32, 32]):
        # self.graph = tf.Graph()
        self.alpha = 0
        # with self.graph.as_default():
        self.regloss = tf.zeros([1])
        self.sy_ob_nn = tf.placeholder(shape=[None, ob_dim], name="ob_nn", dtype=tf.float32)
        self.y = tf.placeholder(shape=[None, 1], name="y", dtype=tf.float32)

        # sy_h1 = lrelu(dense(sy_ob_nn, params[0], "sy_h1", weight_init=normc_initializer(1.0))) # hidden layer
        sy_h = tf.nn.elu(
            self.BN(self.dense(self.sy_ob_nn, params[0], "sh1", weight_init=tf.contrib.layers.xavier_initializer()),
                    params[0]))
        # batch_mean2, batch_var2 = tf.nn.moments(z2_BN,[0])

        for i, l in enumerate(params[1:]):
            # tf.Variable("sy_h"+(i+2)) = lrelu(dense(tf.get_variable("sy_h" + (i+1)), params[0], "h"+l, weight_init=normc_initializer(1.0))) # hidden layer
            sy_h = tf.nn.elu(
                self.BN(self.dense(sy_h, l, "sh" + str(i + 2), weight_init=tf.contrib.layers.xavier_initializer()),
                        l))  # hidden layer

        # self.sy_value = dense(tf.get_variable("sy_h" + len(params)),1, "op", weight_init=normc_initializer(1.0))
        self.sy_value = self.dense(sy_h, 1, "op", weight_init=tf.contrib.layers.xavier_initializer())
        self.loss = tf.reduce_mean(tf.square(self.sy_value - self.y)) + self.alpha * self.regloss
        self.sy_stepsize = tf.placeholder(shape=[],
                                          dtype=tf.float32)  # Symbolic, in case you want to change the stepsize during optimization. (We're not doing that currently)
        # self.optimizer = tf.train.AdamOptimizer(self.sy_stepsize).minimize(self.loss)
        optimizer = tf.train.MomentumOptimizer(self.sy_stepsize, 0.7, use_nesterov=False)
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.sy_stepsize)
        self.gvs = optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in self.gvs]
        self.train_op = optimizer.apply_gradients(capped_gvs)

    def fit(self, X, y, nepoch=100, init_step=1e-3, minibatch_size=64):
        # with self.graph.as_default():
        y = np.reshape(y, (len(y), 1))
        index = np.arange(X.shape[0])
        ploss = 10000.0
        step = init_step
        tau = nepoch / 10.0
        for e in xrange(nepoch):
            curr_index = np.random.choice(index, minibatch_size, replace=True)
            x_batch = X[curr_index, :]
            y_batch = y[curr_index, :]
            loss, gvs = self.sess.run([self.loss, self.gvs], feed_dict={self.sy_ob_nn: X, self.y: y})
            step = init_step * (tau / max(e, tau)) ** 2
            print loss, step, np.sum([np.linalg.norm(grad) for grad, var in gvs])
            # if((loss[0] - ploss) > 0.1 and (e > 500)):
            #	step = step/(1.1)
            # ploss = loss
            self.sess.run([self.train_op], feed_dict={self.sy_ob_nn: x_batch, self.y: y_batch, self.sy_stepsize: step})

    def predict(self, X):
        value = self.sess.run([self.sy_value], feed_dict={self.sy_ob_nn: X})
        # print "value", X.shape, np.shape(value[0])
        return np.squeeze(np.array(value))

    def registerSession(self, sess):
        self.sess = sess

    def BN(self, layer, size):
        batch_mean2, batch_var2 = tf.nn.moments(layer, [0])
        scale2 = tf.Variable(tf.ones([size]))
        beta2 = tf.Variable(tf.zeros([size]))
        return tf.nn.batch_normalization(layer, batch_mean2, batch_var2, beta2, scale2, 1e-8)

    def dense(self, x, size, name, weight_init=None):
        """
        Dense (fully connected) layer

        """
        with tf.variable_scope("ValueNetwork"):
        # with self.graph.as_default():
            w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
            b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
            self.regloss += tf.norm(w)
            return tf.matmul(x, w) + b

    def lrelu(self, x, leak=0.2):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

    #	'''

    # sy_h2 = lrelu(dense(sy_h1, 30, "h2", weight_init=normc_initializer(1.0)))
    # sy_h3 = lrelu(dense(sy_h2, 20, "h3", weight_init=normc_initializer(1.0)))
    #  sy_mean = dense(sy_ob_no, ac_dim, "mean", weight_init=normc_initializer(1.0))
    #		for i,l in enumerate(layers):
    #			sy_h1 = lrelu(dense()
    pass  # YOUR CODE HERE


def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


def gaussian_sample_action(mean, logstdev, a_lb, a_ub):
    #	mean = tf.Print(mean,[mean],"mean is:")

    U = mean + tf.exp(logstdev) * tf.random_normal(tf.shape(mean), mean=0.0, stddev=1.0)

   # U = tf.clip_by_value(U, a_lb, a_ub)
    return U


# def compute_inv(gvs):



def getGaussianKL(mean1, mean2, logstd1, logstd2, n):
    # d = shape[0]

    d = 1
    # std = tf.exp(logstd)
    std1 = tf.exp(logstd1)
    std2 = tf.exp(logstd2)

    #	tr = (std1**2)/(std2**2)

   # mean1 = tf.Print(mean1, [mean1, std1], message="This is Mean1: ")
   # mean2 = tf.Print(mean2, [mean2, std2], message="This is Mean2: ")

    delMean = tf.cast(mean2 - mean1, tf.float32)
    #delMean = tf.Print(delMean, [delMean], message="This is delMean: ")
    p = tf.log(std2 / (std1 + 1e-8)) + ((std1 ** 2) + (delMean) ** 2) / (2 * (std2 ** 2) + 1e-8)
    # p = tf.Print(p, [p, (std1**2 + (delMean)**2)/(2*(std2**2) + 1e-8), tf.log(std2/(std1 + 1e-8)) ], message = "This is p")
    return tf.reduce_mean(p - 0.5)


#	return 0.5*tf.reduce_mean(tf.log((std2**2)/(std1**2)) + tr + tf.multiply(delMean,(delMean/(std2**2))) - d)


def getGaussianDiffEntropy(mean, logstd):
    std = tf.reduce_prod(tf.exp(logstd))
    diffEnt = 0.5 * tf.log(2 * np.pi * np.exp(1) * (std) ** 2)
    return diffEnt


def gaussian_log_prob(mean , logstd, actions):
    dist = tf.contrib.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(logstd))
    logprob = dist.log_prob(actions)
    return logprob


def AddGrad(sy_grad, sy_theta, theta_val):
    #gvs = [(gradvar[0], tf.Print(gradvar[1], [gradvar[1]], "gvs_var")) for i, gradvar in enumerate(gvs)]
   # gvs = optimizer.compute_gradients()

    #gvs_mod = [(sy_grad[i], gradvar[1]) for i, gradvar in enumerate(gvs)]
    #gvs_mod = printList(gvs_mod, "gvs")
    #return optimizer.apply_gradients(gvs_mod)
    #theta_val =printList(theta_val,"thval")
    return ([tf.assign(theta, theta_val[i] + sy_grad[i]) for i, theta in enumerate(sy_theta)])
    #return ([tf.assign(gradvar[1], tf.add(gradvar[1], gradvar[0])) for i, gradvar in enumerate(gvs_mod)])


def main_pendulum(logdir, seed, n_iter, gamma, min_timesteps_per_batch, del_kl, vf_type, vf_params, animate=False):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env = gym.make("Pendulum-v0")
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    logz.configure_output_dir(logdir)
    if vf_type == 'linear':
        print "ln"
        vf = LinearValueFunction(**vf_params)
    elif vf_type == 'nn':
        print "nn"
        vf = NnValueFunction(ob_dim=ob_dim, **vf_params)

    print "bounds calculation"
    a_lb = env.action_space.low
    a_ub = env.action_space.high
    a_bnds = a_ub - a_lb

    print "Symbolic init"
    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)  # batch of observations
    sy_ac_n = tf.placeholder(shape=[None, ac_dim], name="ac",
                             dtype=tf.float32)  # batch of actions taken by the policy, used for policy gradient computation
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)  # advantage function estimate

    with tf.variable_scope("PolicyNetwork"):
        sy_h1 = lrelu(dense(sy_ob_no, 32, "h1", weight_init=tf.contrib.layers.xavier_initializer()))  # hidden layer
    #sy_mean = lrelu(dense(sy_ob_no, ac_dim, "mean", weight_init=tf.contrib.layers.xavier_initializer()))
        sy_h2 = lrelu(dense(sy_h1, 32, "h2", weight_init=tf.contrib.layers.xavier_initializer()))
    #sy_h1 = lrelu(dense(sy_ob_no, 32, "h3", weight_init=normc_initializer(1.0)))

    # sy_h3 = lrelu(dense(sy_h2, 20, "h3", weight_init=normc_initializer(1.0)))
    #sy_mean = dense(sy_h2, ac_dim, "mean", weight_init=tf.contrib.layers.xavier_initializer())
        sy_mean = dense(sy_h2, ac_dim, "mean", weight_init=tf.contrib.layers.xavier_initializer())

    # sy_mean = tf.multiply(tf.cast(a_bnds,tf.float32),tf.tanh(dense(sy_h2, ac_dim, "mean", weight_init=normc_initializer(10.0))))

    sy_logstd = tf.get_variable("logstdev", [ac_dim], initializer=tf.zeros_initializer())  # Variance
    sy_sampled_ac = gaussian_sample_action(sy_mean, sy_logstd, a_lb, a_ub)

    sy_logprob_n = gaussian_log_prob(sy_mean, sy_logstd, sy_ac_n)
    sy_n = tf.shape(sy_ob_no)[0]

    sy_oldmean = tf.placeholder(shape=[None, ac_dim], name='oldmean', dtype=tf.float32)
    sy_oldlogstd = tf.placeholder(shape=[None, ac_dim], name='oldlogstdev', dtype=tf.float32)

    sy_kl = getGaussianKL(sy_oldmean, sy_mean, sy_oldlogstd, sy_logstd, sy_n)
    sy_ent = getGaussianDiffEntropy(sy_mean, sy_logstd)

    sy_surr =  tf.reduce_mean(tf.multiply(sy_adv_n,
                                           sy_logprob_n))  # Loss function that we'll differentiate to get the policy gradient ("surr" is for "surrogate loss")

    dist = tf.contrib.distributions.MultivariateNormalDiag(loc=sy_mean, scale_diag=tf.exp(sy_logstd))
    olddist = tf.contrib.distributions.MultivariateNormalDiag(loc=sy_oldmean, scale_diag=tf.exp(sy_oldlogstd))
    sy_prob = dist.prob(sy_ac_n)
    sy_oldprob = olddist.prob(sy_ac_n)

    sy_lpi = tf.reduce_mean(tf.multiply(sy_adv_n,tf.div(sy_prob,sy_oldprob)))

    sy_stepsize = tf.placeholder(shape=[],
                                 dtype=tf.float32)  # Symbolic, in case you want to change the stepsize during optimization. (We're not doing that currently)
    # update_op = tf.train.AdamOptimizer(sy_stepsize).minimize(sy_surr)
    #optimizer = tf.train.GradientDescentOptimizer(1.0)
    #gvs = optimizer.compute_gradients(sy_surr)
    #vars = tf.trainable_variables()
    #gvs = [(gradvar[0], gradvar[1]) for i, gradvar in enumerate(gvs)]

    # gvs = tf.gradients(sy_surr)
    # var_list = [tf.contrib.layers.flatten(tf.expand_dims(var,0)) for grad,var in gvs]
    # grad_list = [tf.contrib.layers.flatten(tf.expand_dims(grad,0)) for grad,var in gvs]
    #sy_theta = [var for (grad, var) in gvs]
    #sy_gtheta = [grad for (grad, var) in gvs]
    sy_theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "PolicyNetwork")
    sy_gtheta = tf.gradients(sy_surr, sy_theta)


    print "th", sy_theta

    # sy_gtheta, sy_theta = flatten(gvs)
    # sy_theta = tf.placeholder(shape = [None], name = "theta",dtype = tf.float32)
    # sy_gtheta = tf.placeholder(shape = tf.shape(theta), name = "gtheta",dtype = tf.float32)



    sy_kl_test = tf.reduce_mean(tf.contrib.distributions.kl(olddist, dist))
    gradient = tf.gradients(sy_kl, sy_theta)
   # gradient = printList(gradient)
    # print sy_theta.get_shape().as_list(), gradient.get_shape().as_list()
    #gradient = removeNan(gradient)
    #gradient = [tf.clip_by_value(grad, -1., 1.) for grad in gradient]

    sess = tf.Session()

    vf.registerSession(sess)

    vector = [tf.placeholder(dtype=tf.float32, name="vec" + str(i)) for i in xrange(len(sy_theta))]
    # gradvec = [tf.placeholder(dtype = tf.float32,name = gvec + str(i)) for i in xrange(len(theta))]
    gradient_vector_product_test = tf.reduce_sum([tf.reduce_sum(tf.multiply(gradient[i],vector[i])) for i in xrange(len(sy_theta))])
    hessian_vector_product_test = tf.gradients(gradient_vector_product_test, sy_theta)

    def HVP(vec, f_dict):
        temp = {v: d for v, d in zip(vector, vec)}
        f_dict.update(temp)
        return sess.run(hessian_vector_product_test, feed_dict=f_dict)

        # gradient = printList(gradient, "grad")
        # hessian = tf.gradients(gradient, sy_theta)


   # s_unc = CG(sy_kl, sy_theta, sy_gtheta, gradient, 0)
    # s_unc_f = flatten1(s_unc)
   # sy_sunc = [tf.placeholder(dtype = tf.float32, name = "sunc" + str(i)) for i in xrange(len(sy_theta)) ]
   # sy_si = scalarProdList(sy_sunc, tf.sqrt(2 * del_kl / (dotList(sy_sunc, HVP(sy_sunc)))))


    sy_s = [tf.placeholder(dtype=tf.float32, name="s" + str(i)) for i, val in enumerate(sy_theta)]
    # sy_gvs = [(tf.placeholder(dtype = tf.float32), var) for grad, var in gvs]
    sy_grad = [tf.placeholder(dtype=tf.float32, name="grad" + str(i)) for i, val in enumerate(sy_theta)]
    sy_thval = [tf.placeholder(dtype=tf.float32, name="thval" + str(i)) for i, val in enumerate(sy_theta)]


    step_op = AddGrad(sy_grad, sy_theta, sy_thval)

    # gvs_mod = [(tf.placeholder(shape = [None], name = "theta",dtype = tf.float32), tf.placeholder(shape = [None], name = "theta",dtype = tf.float32)) for i in xrange(10)]
    # train_op = optimizer.apply_gradients(gvs_mod)




    # actprob = dist.prob(sy_ac_n)


   # sess.__enter__()  # equivalent to `with sess:`
   # tf.global_variables_initializer().run()  # pylint: disable=E1101
    ##TestCG>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #th = [tf.Variable(np.ones((2, 2))) for _ in xrange(2)]
    #th = tf.Variable(np.random.rand(20), dtype = tf.float32)
    #f = tf.reduce_sum([tf.reduce_sum(tf.square(t)) for t in th])
    #f = tf.reduce_sum(tf.square(th))
    #f = tf.Print(f, [f, th], "f, th")
    #thf = flatten1(th)
    #print thf[0]
   # sess.__exit__()
    #gf = tf.gradients(f, th)
    #print gf, tf.gradients(f, th)


    '''
    hs = []
    for i in xrange(20):
        dfx_i = tf.slice(gf[0], begin=[i], size=[1])
#        hs.append(tf.squeeze(tf.gradients(dfx_i, th)[0]))
        hs.append(tf.gradients(dfx_i, th)[0])


    hess = tf.stack(hs)
    hess = tf.Print(hess, [hess], "hess")
    hs_inv = tf.matrix_inverse(hess)
    yt = tf.placeholder(dtype = tf.float32)
    xt = tf.matmul(hs_inv, tf.reshape(yt, [-1,1]))
    hvp_test = tf.matmul(hess, tf.reshape(yt, [-1,1]))
    ##TestCG>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''

    sess.__enter__()  # equivalent to `with sess:`
    tf.global_variables_initializer().run()  # pylint: disable=E1101
    # vf.registerSession(sess)
    # testShit>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    s_ac = []
    sy_testmean = tf.placeholder(shape=[None, ac_dim], name='oldmean', dtype=tf.float32)
    sy_testlogstd = tf.placeholder(shape=[None, ac_dim], name='oldlogstdev', dtype=tf.float32)
    sy_test_a = tf.placeholder(shape=[None, ac_dim], name='oldlogstdev', dtype=tf.float32)

    sy_sampled_ac_test = gaussian_sample_action(sy_testmean, sy_testlogstd, a_lb, a_ub)
    sy_logprob_test = gaussian_log_prob(sy_testmean, sy_testlogstd, sy_test_a)
    mean_t = np.reshape(np.array([[0.113], [0.111]]), (2, 1))
    std_t = np.reshape(np.array([np.log(0.001)]), (1, 1))
    print a_lb, a_ub

    for i in xrange(1000):
        a = sess.run([sy_sampled_ac_test], feed_dict={sy_testmean: mean_t, sy_testlogstd: std_t})
        s_ac.append(a)
    s_ac = np.array(s_ac)

    print s_ac
    print "std = ", np.std(s_ac, axis=0), "mean= ", np.mean(s_ac, axis=0)
    lp = sess.run([sy_logprob_test],
                  feed_dict={sy_testmean: mean_t, sy_testlogstd: std_t, sy_test_a: np.reshape(s_ac[0], (2, 1))})
    from scipy.stats import norm
    # print s_ac[0][0], s_ac[1][0]
    logprob = np.array([norm.logpdf(s_ac[0][0][0], loc=mean_t[0, :], scale=np.exp(std_t)),
                        norm.logpdf(s_ac[0][0][1], loc=mean_t[1, :], scale=np.exp(std_t))])
    # print "logprob",logprob,"s_ac",s_ac[0][0],"s_ac",s_ac[1][0]
    print "lp_score", np.sum(lp - logprob)
    # testShit>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    ##TestCG>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''
    y = np.random.rand(20)
    #oldx = sess.run(xt, feed_dict={yt: y})
    y_1 = HVPTest([y])
    y_2 = sess.run(hvp_test, feed_dict = {yt: y})
    print "HVPScore", y_1, y_2, np.sum((np.array(y_1) - np.array(y_2).T)**2)
  

   # A = np.array([[4.0,1.0],[1.0,3.0]])
    #y = np.array([1.0,2.0])
    A = np.random.rand(5,5)
    A = np.dot(A.T, A)
    y = np.random.rand(20)

    def Ax(x):
        print "A =", A, x[0].T
        return [np.dot(A, x[0].T)]
    oldx = sess.run(xt, feed_dict = {yt:y})
    #oldx = np.linalg.solve(A,y)
    #newx = CG1(Ax, [y], 10000)
    newx = CG1(HVPTest, [y], 100)
    print "Score = ", newx, oldx, np.sum((oldx - newx) ** 2)
    while True:
        continue
    ##TestCG>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
   '''




    total_timesteps = 0

    stepsize = 0

    for i in range(n_iter):
        print("********** Iteration %i ************" % i)

        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode = (len(paths) == 0 and (i % 10 == 0) and animate)
            while True:
                if animate_this_episode:
                    env.render()
                ob = ob.reshape((1, ob_dim))

                obs.append(ob)
                #                ob = ob.reshape((1,ob_dim))

                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no: np.reshape(ob, (1, ob_dim))})
                acs.append(ac)

                ob, rew, done, _ = env.step(np.clip(ac,-2.0,2.0))
                rewards.append(rew)
                if done:
                    break
                #    print "path", len(obs), np.array(obs)
            path = {"observation": np.squeeze(np.array(obs)), "terminated": terminated,
                    "reward": np.squeeze(np.array(rewards)), "action": np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch
        # Estimate advantage function
        vtargs, vpreds, advs = [], [], []
        for path in paths:
            rew_t = path["reward"]
            return_t = discount(rew_t, gamma)
            vpred_t = vf.predict(path["observation"])
            #         print vpred_t,  return_t.shape
            adv_t = return_t - vpred_t
            # adv_t = (return_t - return_t.mean())/(return_t.std() + 1e-08)
            # adv_t = return_t
            #     print adv_t.shape
            advs.append(adv_t)
            vtargs.append(return_t)
            vpreds.append(vpred_t)

        # Build arrays for policy update
        #    print len(advs), advs[0]
        ob_no = np.concatenate([path["observation"] for path in paths])

        ac_n = np.concatenate([path["action"] for path in paths])
        adv_n = np.concatenate(advs)
        #   print adv_n.shape

        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        vtarg_n = np.concatenate(vtargs)
        vpred_n = np.concatenate(vpreds)
        #   print "Here2"
        #   print ob_no.shape
        ob_no = np.squeeze(ob_no)
        ac_n = np.reshape(ac_n, (ac_n.shape[0], ac_dim))
        vf.fit(ob_no, vtarg_n)
        # sh=sy_mean.get_shape().as_list()
        # print sh
        # oldoldlogstd = 1.01


        #        sy_surr = tf.Print(sy_surr,[sy_surr],"Loss =")
        #print "Loss = ", sess.run(sy_surr, feed_dict={sy_ob_no: ob_no, sy_ac_n: ac_n, sy_adv_n: standardized_adv_n,
        #                                              sy_stepsize: stepsize})
        # Policy update
        oldmean, oldlogstd, oldlogprob = sess.run([sy_mean, sy_logstd, sy_logprob_n],
                                                  feed_dict={sy_ob_no: ob_no, sy_ac_n: ac_n,
                                                             sy_adv_n: standardized_adv_n, sy_stepsize: stepsize})
        #olds = sess.run([s],
        #                feed_dict={sy_ob_no: ob_no, sy_ac_n: ac_n, sy_adv_n: standardized_adv_n, sy_oldmean: oldmean,
        #                           sy_oldlogstd: np.reshape(oldlogstd, [1, ac_dim])})
        f_dict = {sy_ob_no: ob_no, sy_ac_n: ac_n, sy_adv_n: standardized_adv_n, sy_oldmean: oldmean,
                     sy_oldlogstd: np.reshape(oldlogstd, [1, ac_dim])}
        oldtheta, oldgtheta, oldkl, oldkltest = sess.run([sy_theta, sy_gtheta, sy_kl, sy_kl_test],
                           feed_dict=f_dict)
       # print "old_kl", oldkl
        ##>>>>>>>>>>>>Check Positive Def
        '''
        hessian = []
        for i,th1 in enumerate(oldtheta):
             for j in xrange(np.size(th1)):
                zmat = [np.zeros_like(th2) for th2 in oldtheta]
                temp = np.zeros(np.size(th1))
                temp[j] = 1.0
                zmat[i] = np.reshape(temp, th1.shape)
                #print zmat
                hvp = HVP(zmat, f_dict)
                hessian.append(np.concatenate(tuple([np.ravel(h) for h in hvp])))
       # print hessian
        def is_pos_def(x):
            return np.all(np.linalg.eigvals(x) >= 0)
        print "Positive Def", is_pos_def(np.array(hessian))
        ev = np.linalg.eigvals(hessian)
        print "Eigen Hessian", np.sum(ev[ev < 0])
        '''
        ##>>>>>>>>>>>>

        s_unc = CG(HVP, oldgtheta, f_dict, 100)
       # olds = sess.run(sy_si, feed_dict = {i: np.asarray(d, dtype=np.float32) for i, d in zip(sy_sunc, s_unc)})
        print s_unc
        olds = scalarProdList(s_unc, np.sqrt(2 * del_kl / (dotProdList(s_unc, HVP(s_unc, f_dict)) + 1e-9)))
        #print "olds", olds
        ##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> LineSearch
        # train_op = optimizer.apply_gradients(gvs)
        #dict_s = {i: np.asarray(d, dtype=np.float32) for i, d in zip(sy_s, olds)}
        #dict_g = {i: np.asarray(d, dtype=np.float32) for i, d in zip(sy_grad, )}
        dict_th = {i: np.asarray(d, dtype=np.float32) for i, d in zip(sy_thval, oldtheta)}


        #f = {sy_ob_no: ob_no, sy_ac_n: ac_n, sy_adv_n: standardized_adv_n, sy_oldmean: oldmean,
        #     sy_oldlogstd: np.reshape(oldlogstd, [1, ac_dim])}
        #f.update(dict_s)
        #dict_s.update(f_dict)
        #dict_g.update(f_dict)
        #dict_g.update(dict_th)
        currs = copy.deepcopy(olds)
       # sess.run(step_op, feed_dict=dict_g)
        old_obj = sess.run(sy_lpi, feed_dict=f_dict)
        count = 0.0
        while True:
            count += 2.0
            print "count =", count
            # gvs_mod = [(tf.multiply(s[i],gradvar[0]),gradvar[1]) for i,gradvar in enumerate(gvs)]
            #gtheta = Mul2List(oldgtheta, currs)
            gtheta = currs
            dict_g = {i: np.asarray(d, dtype=np.float32) for i, d in zip(sy_grad, gtheta)}
            dict_g.update(f_dict)
            dict_g.update(dict_th)
           # f = {sy_ob_no: ob_no, sy_ac_n: ac_n, sy_adv_n: standardized_adv_n, sy_oldmean: oldmean,
           #      sy_oldlogstd: np.reshape(oldlogstd, [1, ac_dim])}
           # f.update(dict_s)
            # train_op = optimizer.apply_gradients(gvs_mod)


            sess.run(step_op, feed_dict=dict_g)
            #ngtheta, ntheta = sess.run([sy_gtheta, sy_theta], feed_dict = f_dict)
            #dtheta = Sub2List(ntheta, oldtheta)
            #print "Score", SumAList(Sub2List(dtheta, [np.array(s) / count for i, s in enumerate(currs)]))
            #oldtheta = ntheta
            curr_kl, curr_kl_test = sess.run([sy_kl, sy_kl_test], feed_dict=f_dict)
            print "curr_kl", curr_kl,"curr_kl_test",curr_kl_test,"del_kl", del_kl

            if (curr_kl > del_kl):
                currs = scalarProdList(olds, 1/count)
                continue
            curr_obj = sess.run(sy_lpi, feed_dict=f_dict)
            print "curr_obj =", curr_obj, "old_obj =", old_obj
            if (curr_obj >= old_obj):
                break
            else:
                currs = scalarProdList(olds, 1/count)

        ##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        kl, ent = sess.run([sy_kl, sy_ent], feed_dict={sy_ob_no: ob_no, sy_oldmean: oldmean,
                                                       sy_oldlogstd: np.reshape(oldlogstd, [1, ac_dim])})

        print "Diag"
        # Log diagnostics
        logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logz.log_tabular("KLOldNew", kl)
        logz.log_tabular("Entropy", ent)
        logz.log_tabular("EVBefore", explained_variance_1d(vpred_n, vtarg_n))
        logz.log_tabular("EVAfter", explained_variance_1d(vf.predict(ob_no), vtarg_n))
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
        logz.dump_tabular()


def main_pendulum1(d):
    return main_pendulum(**d)


if __name__ == "__main__":
    if 0:
        main_cartpole(logdir=None)  # when you want to start collecting results, set the logdir

    if 1:
        general_params = dict(gamma=0.97, animate=False, min_timesteps_per_batch=2500, n_iter=1000)
        params = [

            dict(logdir=None, seed=0, desired_kl=2e-6, vf_type='linear', vf_params={}, **general_params),

            #           dict(logdir=  '/tmp/ref11/linearvf-kl2e-3-seed0' , seed=0, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
            #            dict(logdir=  '/tmp/ref0/nnvf-kl2e-3-seed0' , seed=0, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
            #            dict(logdir=  '/tmp/ref0/linearvf-kl2e-3-seed1' , seed=1, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
            #            dict(logdir=  '/tmp/ref0/nnvf-kl2e-3-seed1' , seed=1, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
            #            dict(logdir=  '/tmp/ref0/linearvf-kl2e-3-seed2' , seed=2, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
            #            dict(logdir=  '/tmp/ref0/nnvf-kl2e-3-seed2' , seed=2, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
        ]
        p = dict(logdir=None, seed=0, desired_kl=2e-6, vf_type='linear', vf_params={}, **general_params),

        main_pendulum(logdir=None, seed=0, del_kl=2e-4, vf_type='nn', vf_params={}, gamma=0.97, animate=False,
                      min_timesteps_per_batch=2500, n_iter=1000)
# import multiprocessing

#        p = multiprocessing.Pool()
#        p.map(main_pendulum1, params)
