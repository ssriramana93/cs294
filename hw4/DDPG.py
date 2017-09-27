import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
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


def dense(x, size, name, weight_init=None,regularizer=None, scope = None):
    """
    Dense (fully connected) layer
    """
    with tf.variable_scope(scope):
        w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init, regularizer=regularizer)
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


def gaussian_log_prob(mean, logstdev, ac_taken):
    dist = tf.contrib.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(logstdev))
    logprob = dist.log_prob(ac_taken)
    logprob = tf.Print(logprob, [logprob], message="This is LogProb: ")

    return logprob


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


class CriticNetwork(object):
    #	'''
    def __init__(self, ob_dim=10, ac_dim=10, params=[128, 128], scope = "Default"):
        self.scope = scope
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.minibatch_size = 64
        with tf.variable_scope(scope):
            self.alpha = 1e-2

        #self.regloss = tf.zeros([1])
            #self.sy_ob_no_p = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
            #self.sy_ac_n_p = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32)
            with tf.variable_scope(scope + "special"):
                self.sy_ob_no = tf.Variable(np.ones((self.minibatch_size, ob_dim)), name="ob", dtype=tf.float32)
                self.sy_ac_n = tf.Variable(np.ones((self.minibatch_size, ac_dim)), name="ac", dtype=tf.float32)
            #self.sy_ac_v = tf.variable(np.ones((1, ac_dim)), name="ac", dtype=tf.float32)

           # self.sy_ob_ac_n = tf.concat([self.sy_ob_no, self.sy_ac_n], axis = 1)
            with tf.variable_scope(scope + "trainable"):
                scope = scope + "trainable"

                self.sy_y = tf.placeholder(shape=[None, 1], name="qtrue", dtype=tf.float32)
                sy_h1 = tf.nn.elu(dense(self.sy_ob_no, params[0], "hs", weight_init=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(scale = self.alpha),scope = scope))
                self.sy_h1_ac_n = tf.concat([sy_h1, self.sy_ac_n], axis=1)
                sy_h = tf.nn.elu(dense(self.sy_h1_ac_n, params[0], "h1", weight_init=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(scale = self.alpha),scope = scope))

           # sy_h = tf.nn.elu(dense(self.sy_ob_ac_n, params[0], "h1", weight_init=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(scale = self.alpha),scope = scope))
          #  sy_h = self.sy_ob_ac_n
                for i, l in enumerate(params[1:]):
                    sy_h = tf.nn.elu(dense(sy_h, l, "h" + str(i + 2), weight_init=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(scale = self.alpha), scope = scope))  # hidden layer

        # self.sy_value = dense(tf.get_variable("sy_h" + len(params)),1, "op", weight_init=normc_initializer(1.0))
                self.sy_qvalue = dense(sy_h, 1, "qvalue", weight_init=normc_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale = self.alpha), scope = scope)
                self.loss = tf.reduce_mean(tf.square(self.sy_qvalue - self.sy_y)) + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope))
                self.sy_stepsize = tf.placeholder(shape=[],
                                              dtype=tf.float32)  # Symbolic, in case you want to change the stepsize during optimization. (We're not doing that currently)
            # self.optimizer = tf.train.AdamOptimizer(self.sy_stepsize).minimize(self.loss)
                optimizer = tf.train.AdamOptimizer(self.sy_stepsize)
            # optimizer = tf.train.AdamOptimizer(learning_rate=self.sy_stepsize)
                self.gvs = optimizer.compute_gradients(self.loss)
                capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in self.gvs if grad != None]
                self.train_op = optimizer.apply_gradients(capped_gvs)
                self.tau = 0.7

    def AssignOp(self, obs, action):
           return [tf.assign(self.sy_ob_no, obs), tf.assign(self.sy_ac_n, action)]



    def fit(self, X, y, nepoch=10, init_step=5e-5, minibatch_size=64):
        # with self.graph.as_default():
        y = np.reshape(y, (len(y), 1))
        index = np.arange(X.shape[0])
        #ploss = 10000.0
        step = init_step
        tau = nepoch / 10.0
       # print ("start")
        for e in range(nepoch):
            curr_index = np.random.choice(index, minibatch_size, replace=True)
            x_batch = X[curr_index, :]
            y_batch = y[curr_index, :]
           # loss, gvs = self.sess.run([self.loss, self.gvs], feed_dict={self.sy_ob_no: x_batch[:,:self.ob_dim],self.sy_ac_n: x_batch[:,(self.ob_dim):], self.sy_y: y})
            #loss = self.sess.run(self.loss, feed_dict={self.sy_ob_no: x_batch[:,:self.ob_dim],self.sy_ac_n: x_batch[:,(self.ob_dim):], self.sy_y: y_batch, self.sy_stepsize: step})
          #  print (loss)
            step = init_step * (tau / max(e, tau))
           # print loss, step, np.sum([np.linalg.norm(grad) for grad, var in gvs])
            # if((loss[0] - ploss) > 0.1 and (e > 500)):
            #	step = step/(1.1)
            # ploss = loss
            assign_op = self.AssignOp(x_batch[:,:self.ob_dim].reshape(-1, self.ob_dim),x_batch[:,(self.ob_dim):].reshape(-1, self.ac_dim))
            self.sess.run(assign_op)
            _, loss = self.sess.run([self.train_op,self.loss], feed_dict = {self.sy_stepsize: step, self.sy_y: y_batch.reshape(-1,1)})
            print (loss)

#            for i in range(minibatch_size):
#                assign_op = self.AssignOp(x_batch[i,:self.ob_dim].reshape(-1, self.ob_dim),x_batch[i,(self.ob_dim):].reshape(-1, self.ac_dim))
#                self.sess.run(assign_op)
#                _, loss = self.sess.run([self.train_op,self.loss], feed_dict = {self.sy_stepsize: step, self.sy_y: y_batch[i].reshape(1,1)})

                #print (loss)
            #self.sess.run([self.train_op], feed_dict={self.sy_ob_no: x_batch[:,:self.ob_dim],self.sy_ac_n: x_batch[:,(self.ob_dim):], self.sy_y: y_batch, self.sy_stepsize: step})

    def predict(self, X):
        value = []

        #value = self.sess.run(self.sy_qvalue, feed_dict={self.sy_ob_no: X[:,:self.ob_dim],self.sy_ac_n: X[:,(self.ob_dim):]})

        #for i in range(X.shape[0] % self.minibatch_size):
        assign_op = self.AssignOp(X[:,:self.ob_dim].reshape(-1, self.ob_dim),X[:,(self.ob_dim):].reshape(-1, self.ac_dim))

#        assign_op = self.AssignOp(X[i*self.minibatch_size: 2*i*self.minibatch_size,:self.ob_dim].reshape(-1, self.ob_dim),X[i*self.minibatch_size: 2*i*self.minibatch_size,(self.ob_dim):].reshape(-1, self.ac_dim))
        self.sess.run(assign_op)
        y = self.sess.run(self.sy_qvalue)
        value.append(y.reshape(-1))

        return np.asarray(value)

    def registerSession(self, sess):
        self.sess = sess

    def BN(self, layer, size):
        batch_mean2, batch_var2 = tf.nn.moments(layer, [0])
        scale2 = tf.Variable(tf.ones([size]))
        beta2 = tf.Variable(tf.zeros([size]))
        return tf.nn.batch_normalization(layer, batch_mean2, batch_var2, beta2, scale2, 1e-8)


    def UpdateParams(self, values, tau):
        return [tf.assign(var, tau*values[i] + (1 - tau)*self.sess.run(var)) for i, var in enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope))]

    def GetParams(self):
        values = []
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope):
            values.append(self.sess.run(var))
        return values


    def GetGradientA(self):
        return tf.gradients(-self.sy_qvalue, self.sy_ac_n)
    '''
    def dense(self, x, size, name, weight_init=None):
        """
        Dense (fully connected) layer

        """
        # with self.graph.as_default():
        w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
        b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
        self.regloss += tf.norm(w)
        return tf.matmul(x, w) + b

    def lrelu(self, x, leak=0.2):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

	'''

    # sy_h2 = lrelu(dense(sy_h1, 30, "h2", weight_init=normc_initializer(1.0)))
    # sy_h3 = lrelu(dense(sy_h2, 20, "h3", weight_init=normc_initializer(1.0)))
    #  sy_mean = dense(sy_ob_no, ac_dim, "mean", weight_init=normc_initializer(1.0))
    #		for i,l in enumerate(layers):
    #			sy_h1 = lrelu(dense()
      # YOUR CODE HERE

def BN(layer, size):
        batch_mean2, batch_var2 = tf.nn.moments(layer, [0])
        scale2 = tf.Variable(tf.ones([size]))
        beta2 = tf.Variable(tf.zeros([size]))
        return tf.nn.batch_normalization(layer, batch_mean2, batch_var2, beta2, scale2, 1e-8)



def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


def gaussian_sample_action(mean, logstdev, a_lb, a_ub):
    #	mean = tf.Print(mean,[mean],"mean is:")

    U = mean + tf.exp(logstdev) * tf.random_normal(tf.shape(mean), mean=0.0, stddev=1.0)
    # U = tf.clip_by_value(U,a_lb,a_ub)
    #	U = tf.Print(U,[U],"U is:")
    # lb_index = U < a_lb
    # ub_index = U > a_ub
    #	print lb_index.shape, ub_index.shape

    # ub = lb = tf.ones(tf.shape(U))
    # lb = a_lb*lb
    # ub = a_ub*ub
    # lb = tf.reshape(np.ones()a_lb,tf.shape(U))
    # ub = tf.reshape(a_ub,tf.shape(U))
    # lb = a_lb
    # ub = a_ub
    # U = tf.where(lb_index,U,lb)
    # U = tf.where(ub_index,U,ub)
    # U = tf.where(lb_index, tf.cast(lb,tf.float32), U)
    # U = tf.where(ub_index, tf.cast(ub,tf.float32), U)
    return U


def getGaussianKL(mean1, mean2, logstd1, logstd2, n):
    # d = shape[0]

    d = 1
    # std = tf.exp(logstd)
    std1 = tf.exp(logstd1)
    std2 = tf.exp(logstd2)

    #	tr = (std1**2)/(std2**2)

    mean1 = tf.Print(mean1, [mean1, std1], message="This is Mean1: ")
    mean2 = tf.Print(mean2, [mean2, std2], message="This is Mean2: ")

    delMean = tf.cast(mean2 - mean1, tf.float32)
    delMean = tf.Print(delMean, [delMean], message="This is delMean: ")
    p = tf.log(std2 / (std1 + 1e-8)) + ((std1 ** 2) + (delMean) ** 2) / (2 * (std2 ** 2) + 1e-8)
    # p = tf.Print(p, [p, (std1**2 + (delMean)**2)/(2*(std2**2) + 1e-8), tf.log(std2/(std1 + 1e-8)) ], message = "This is p")
    return tf.reduce_mean(p - 0.5)


#	return 0.5*tf.reduce_mean(tf.log((std2**2)/(std1**2)) + tr + tf.multiply(delMean,(delMean/(std2**2))) - d)


def getGaussianDiffEntropy(mean, logstd):
    std = tf.reduce_prod(tf.exp(logstd))
    diffEnt = 0.5 * tf.log(2 * np.pi * np.exp(1) * (std) ** 2)
    return diffEnt

def SampleFromReplayBuffer(replay_buffer, N=100, maxLen=1e+6):

    if (len(replay_buffer) > maxLen):
        del replay_buffer[0]
    length = len(replay_buffer)
    index = np.random.choice(length, length, replace = 'True')
    if (length >= N):
        index = np.random.choice(length, N, replace = 'True')

    ob_no = np.array([replay_buffer[i][0] for i in index])
    ac_n = np.array([replay_buffer[i][1] for i in index])
    r_n = np.array([replay_buffer[i][2] for i in index])
    ob_next = np.array([replay_buffer[i][3] for i in index])
    t_batch = np.array([replay_buffer[i][4] for i in index])

    return [ob_no, ac_n, r_n, ob_next, t_batch]
    #return np.array([np.concatenate([replay_buffer[i][0], replay_buffer[i][1], replay_buffer[i][2], replay_buffer[i][3]]) for i in index])



def main_pendulum(logdir, seed, n_iter, gamma, min_timesteps_per_batch, initial_stepsize, desired_kl, vf_type,
                  vf_params, tau = 0.01, animate=False):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env = gym.make("Pendulum-v0")
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    logz.configure_output_dir(logdir)
    critic = CriticNetwork(ob_dim, ac_dim, scope = "CriticNetwork")
    critic_target = CriticNetwork(ob_dim, ac_dim, scope = "ACriticNetworkTarget")

    replay_buffer = []


    print ("bounds calculation")
    a_lb = env.action_space.low
    a_ub = env.action_space.high
    a_bnds = a_ub - a_lb

    print ("Symbolic init")
    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)  # batch of observations
    sy_ob_next = tf.placeholder(shape=[None, ob_dim], name="ob_next", dtype=tf.float32)  # batch of observations

    sy_ac_n = tf.placeholder(shape=[None, ac_dim], name="ac",
                             dtype=tf.float32)  # batch of actions taken by the policy, used for policy gradient computation
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)  # advantage function estimate
    sy_r_n = tf.placeholder(shape = [None, 1], name = "rew", dtype = tf.float32)

    with tf.variable_scope("ActorNetwork"):
        actor_alpha = 1e-2
        sy_h1 = lrelu(BN(dense(sy_ob_no, 128, "h1", weight_init=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(scale = actor_alpha), scope = "ActorNetwork"), 128))  # hidden layer
        sy_h2 = lrelu(BN(dense(sy_h1, 128, "h2", weight_init=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(scale = actor_alpha), scope = "ActorNetwork"),128))
        sy_h3 = lrelu(BN(dense(sy_h2, 128, "h3", weight_init=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(scale = actor_alpha),scope = "ActorNetwork"), 128))

#        sy_mean = tf.tanh(dense(sy_h3, ac_dim, "mean", weight_init=tf.contrib.layers.xavier_initializer(), scope = "ActorNetwork"))*2
        sy_mean = dense(sy_h3, ac_dim, "mean", weight_init=normc_initializer(),regularizer=tf.contrib.layers.l2_regularizer(scale = actor_alpha), scope = "ActorNetwork")


    with tf.variable_scope("NActorNetworkTarget"):
        sy_h1_t = lrelu(BN(dense(sy_ob_no, 128, "h1", weight_init=tf.contrib.layers.xavier_initializer(),scope = "NActorNetworkTarget"),128))  # hidden layer
        sy_h2_t = lrelu(BN(dense(sy_h1_t, 128, "h2", weight_init=tf.contrib.layers.xavier_initializer(),scope ="NActorNetworkTarget"),128))
        sy_h3_t = lrelu(BN(dense(sy_h2_t, 128, "h3", weight_init=tf.contrib.layers.xavier_initializer(),scope ="NActorNetworkTarget"), 128))
#        sy_mean_t = tf.tanh(dense(sy_h3_t, ac_dim, "mean", weight_init=tf.contrib.layers.xavier_initializer(), scope = "NActorNetworkTarget"))*2
        sy_mean_t = dense(sy_h3_t, ac_dim, "mean", weight_init=normc_initializer(), scope = "NActorNetworkTarget")


    #sy_logstd = tf.get_variable("logstdev", sy_ob_ac_p[ac_dim], initializer=tf.zeros_initializer())  # Variance
    #sy_logstd = tf.constant(np.ones(ac_dim)*np.exp(2.0), dtype = tf.float32)
    sy_logstd = tf.placeholder(dtype = tf.float32)
    sigma = 2.0
    sigma_decay = 0.97
    sy_sampled_ac = gaussian_sample_action(sy_mean, sy_logstd, a_lb, a_ub)

    #sy_logprob_n = gaussian_log_prob(sy_mean, logstd, sy_ac_n)
    sy_n = tf.shape(sy_ob_no)[0]
    #sy_N = tf.constant(100, dtype = tf.int32)
    #sy_surr = -tf.reduce_mean(tf.multiply(sy_adv_n,
    #                                      sy_logprob_n))
    print (tf.slice(sy_mean, [0,0], [0, 1]))
    sy_critic_g = critic.GetGradientA()
    vars_actor = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = "ActorNetwork")
    vars_actor_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = "NActorNetworkTarget")

    vars_critic = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = "CriticNetwork")
    vars_critic_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = "NCriticNetworkTarget")


    #print sy_critic_g, tf.gradients(sy_mean, vars)
    #sy_actor_g = [tf.gradients(tf.slice(sy_mean, [0,0], [1, (i + 1)]), vars_actor) for i in range(ac_dim)]
    #sy_ob = tf.placeholder(shape=[None, ob_dim], dtype = tf.float32)
    assign_op = critic.AssignOp(sy_ob_no, sy_mean)

    #sy_actor_g = tf.gradients(sy_mean, vars_actor, sy_critic_g)
    #critic_g = tf.Placeholder(dtype = tf.float32)
    #sy_actor_loss = tf.reduce_mean(tf.matmul(sy_mean,critic_g) + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope = "ActorNetwork"))
    sy_actor_loss = -tf.reduce_mean(critic.sy_qvalue) + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope = "ActorNetwork"))



    #sy_grad = [tf.add_n([tf.squeeze(tf.slice(sy_critic_g[i], [0, 0], [1 , i+1]))*sy_actor_g[i][j] for i in range(ac_dim)]) for j in range(len(sy_actor_g[0]))]



    sy_g = [tf.placeholder(dtype = tf.float32) for _ in range(len(vars_actor))]
    grad_var = zip(sy_g, vars_actor)

    update_op = tf.train.AdamOptimizer(5e-6).minimize(sy_actor_loss, var_list = vars_actor)

    #update_op = optimizer.
    #update_op = optimizer.apply_gradients(grad_var)

    #tau = 0.8
    sess = tf.Session()
    sy_tau = tf.placeholder(dtype = tf.float32)
    actor_update = [tf.assign(var2, sy_tau * var1 + (1 - sy_tau) * var2) for var1, var2 in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "ActorNetwork"), tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "NActorNetworkTarget"))]

   
    print (vars_critic)



    '''
    sy_oldmean = tf.placeholder(shape=[None, ac_dim], name='oldmean', dtype=tf.float32)
    sy_oldlogstd = tf.placeholder(shape=[None, ac_dim], name='oldlogstdev', dtype=tf.float32)
    sy_kl = getGaussianKL(sy_oldmean, sy_mean, sy_oldlogstd, sy_logstd, sy_n)
    sy_ent = getGaussianDiffEntropy(sy_mean, sy_logstd)
    '''


    sess.__enter__()  # equivalent to `with sess:`
    tf.global_variables_initializer().run()  # pylint: disable=E1101
    critic.registerSession(sess)
    critic_target.registerSession(sess)



    total_timesteps = 0
    stepsize = initial_stepsize
    params = critic.GetParams()
    sess.run(critic_target.UpdateParams(params, 1.0))
    sess.run(actor_update, feed_dict={sy_tau:1.0})

    n_update = 5
    import time
    N = 64
    for i in range(n_iter):
        print("********** Iteration %i ************" % i)
        '''
        print "critic_params", critic.GetParams()
        print "critic_target", critic_target.GetParams()
        print "actor_params", sess.run(vars_actor)
        print "actor_target_params", sess.run(vars_actor_target)
        '''
        timesteps_this_batch = 0
        paths = []

        while True:
            sigma = sigma * sigma_decay
            if (sigma < 1e-3):
                sigma = 1e-3
            ob = env.reset()
            print ("reseted", sigma)
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode = (len(paths) == 0 and (i % 10 == 0) and animate)
            while True:
               
                if animate_this_episode:
                    env.render()
                ob = ob.reshape((1, ob_dim))

                obs.append(ob)
                #                ob = ob.reshape((1,ob_dim))
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no: np.reshape(ob, (1, ob_dim)), sy_logstd: np.log(sigma)})
                ac = np.clip(ac, -2.0, 2.0)
                #print (ac)
                acs.append(ac)

                ob_next, rew, done, _ = env.step(ac)
                rewards.append(rew)
                if(done):
                    terminated = True
                replay_buffer.append([ob, ac, rew, ob_next, terminated])
                ob = ob_next

                if (len(replay_buffer) < N):
                    if (done):
                        break

                    continue


                '''    
                ob_no, ac_n, r_n, ob_next_no = SampleFromReplayBuffer(replay_buffer, N)
                
                ob_no = np.reshape(ob_no, (-1,ob_dim))
                ac_n = np.reshape(ac_n, (-1,ac_dim))
                r_n = np.reshape(ac_n, (-1,ac_dim))
                ob_next_no = np.reshape(ob_next_no, (-1,ob_dim))


                ob_ac_n = np.concatenate([ob_no, ac_n], axis = 1)
                qsa_next = sess.run(sy_mean_t, feed_dict= {sy_ob_no: ob_next_no})
                ob_ac_next = np.concatenate([ob_no, qsa_next], axis = 1)
                y = r_n + gamma*critic_target.predict(ob_ac_next)
                critic.fit(ob_ac_n,y)
                critic_g = sess.run(sy_critic_g, feed_dict={sy_ob_no: ob_no, sy_ac_n: ac_n, sy_r_n: r_n, critic.sy_ac_n: ac_n, critic.sy_ob_no: ob_no})

                sum = []
                for i in range(N):
                    lp = sess.run(sy_actor_g, feed_dict={sy_ob_no: ob_no[i,:].reshape((-1,ob_dim)), sy_ac_n: ac_n[i,:].reshape((-1,ac_dim)), sy_r_n: r_n[i,:].reshape((-1,1)), critic.sy_ac_n: ac_n[i,:].reshape((-1,ac_dim)), critic.sy_ob_no: ob_no[i,:].reshape((-1,ob_dim))})
                    grad = [np.sum([critic_g[a][i]*lp[a][j] for a in range(ac_dim)], axis = 0) for j in range(len(lp[0]))]

                    if  sum == []:
                        sum = grad
                        continue
                    #print sess.run(sy_critic_g, feed_dict={sy_ob_no: ob_no, sy_ac_n: ac_n, sy_r_n: r_n, critic.sy_ac_n: ac_n, critic.sy_ob_no: ob_no})

                    sum = [grad1 + grad2 for grad1, grad2 in zip(sum, grad)]

                sum = [grad/N for grad in sum]
                fd = {i: d for i, d in zip(sy_g, sum)}
                f1 = {sy_ob_no: ob_no, sy_ac_n: ac_n, sy_r_n: r_n, critic.sy_ac_n: ac_n, critic.sy_ob_no: ob_no}
                f1.update(fd)

                sess.run(update_op, feed_dict = f1)
                params = critic.GetParams()
                critic_target.UpdateParams(params)
                sess.run(actor_update)
                '''



        #sess.run(update_op, feed_dict=fd)
                
                
                if(done):
                    break
                    #    print "path", len(obs), np.array(obs)
            path = {"observation": np.squeeze(np.array(obs)), "terminated": terminated,
                    "reward": np.squeeze(np.array(rewards)), "action": np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            total_timesteps += timesteps_this_batch

           # break
            # print timesteps_this_batch
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        # Estimate advantage function
        #'''


            # Build arrays for policy update
            #    print len(advs), advs[0]

        '''
        index =
        ob_no = #np.concatenate([path["observation"] for path in paths])

        ac_n = np.concatenate([path["action"] for path in paths])
        adv_n = np.concatenate(advs)
        r_n -
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
        print "Loss = ", sess.run(sy_surr, feed_dict={sy_ob_no: ob_no, sy_ac_n: ac_n, sy_adv_n: standardized_adv_n,
                                                      sy_stepsize: stepsize})
        # Policy update
        oldmean, oldlogstd, oldlogprob = sess.run([sy_mean, sy_logstd, sy_logprob_n],
                                                  feed_dict={sy_ob_no: ob_no, sy_ac_n: ac_n,
                                                             sy_adv_n: standardized_adv_n, sy_stepsize: stepsize})
        sess.run([update_op],
                 feed_dict={sy_ob_no: ob_no, sy_ac_n: ac_n, sy_adv_n: standardized_adv_n, sy_stepsize: stepsize})

        # print "ac = ", ac_n, "oldmean = ", oldmean,"oldlogstd = ", oldlogstd
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        std1 = np.exp(oldoldlogstd)
        std2 = np.exp(oldlogstd)
        tr = (std1**2)/(std2**2)
        delMean = oldoldmean - oldmean
        print 0.5*np.mean(np.log((std2**2)/(std1**2)) + tr + np.multiply(delMean,delMean/(std2**2)) - 1)
        oldoldmean = oldmean
        oldoldlogstd = oldlogstd
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        kl, ent = sess.run([sy_kl, sy_ent], feed_dict={sy_ob_no: ob_no, sy_oldmean: oldmean,
                                                       sy_oldlogstd: np.reshape(oldlogstd, [1, ac_dim])})
        # oldoldmean = oldmean
        # if (n_iter > 150):
        # 	desired_kl = 1e-5

        print "Done"
        if kl > desired_kl * 2:
            stepsize /= 1.5
            print('stepsize -> %s' % stepsize)
        elif kl < desired_kl / 2:
            stepsize *= 1.5
            print('stepsize -> %s' % stepsize)
        else:
            print('stepsize OK')

        print "Diag"
        
        '''

        for _ in range(n_update):
            ob_no, ac_n, r_n, ob_next_no, t_batch = SampleFromReplayBuffer(replay_buffer, N)


            ob_no = np.reshape(ob_no, (-1, ob_dim))
            ac_n = np.reshape(ac_n, (-1, ac_dim))
            r_n = np.reshape(r_n, (-1,1))
            ob_next_no = np.reshape(ob_next_no, (-1, ob_dim))
            t_batch = np.reshape(t_batch, (-1, 1))

            ##All Fitings
            ob_ac_n = np.concatenate([ob_no, ac_n], axis=1)
            ac_next = sess.run(sy_mean_t, feed_dict={sy_ob_no: ob_next_no})
            ob_ac_next = np.concatenate([ob_next_no, ac_next], axis=1)
            q_next = critic_target.predict(ob_ac_next)
            y = r_n + gamma *q_next.reshape((-1,1))
            critic.fit(ob_ac_n, y)



            ##All grad calculations
            #actor_mean = sess.run(sy_mean, feed_dict={sy_ob_no: ob_no})
            # critic_g = sess.run(sy_critic_g, feed_dict={critic.sy_ac_n: actor_mean,
            #                                            critic.sy_ob_no: ob_no})
            sm = []
            # critic_g = sess.run(sy_critic_g, feed_dict={critic.sy_ac_n: actor_mean,
            #                                            critic.sy_ob_no: ob_no})
            length = len(ob_no)
            sess.run(assign_op, feed_dict = {sy_ob_no: ob_no})

            for _ in range(10*i):
                sess.run(update_op)
                print ("Actor Loss", sess.run(sy_actor_loss))
            #for i in range(length%32):
                #sess.run(assign_op, feed_dict = {sy_ob: ob_no[i,:].reshape((-1,ob_dim)), sy_ob_no: ob_no[i,:].reshape((-1,ob_dim))})

                #sess.run(update_op)
            
            #print (length)
            '''
            for i in range(length):
            lp = sess.run(sy_actor_g, feed_dict={sy_ob_no: ob_no[i, :].reshape((-1, ob_dim)),
                                                 critic.sy_ac_n: actor_mean[i, :].reshape((-1, ac_dim)),
                                                 critic.sy_ob_no: ob_no[i, :].reshape((-1, ob_dim))})


            # print "critic_g", lp
            #                                            critic.sy_ob_no: ob_no})

            grad = lp
            # grad = [np.sum([critic_g[a][i] * lp[a][j] for a in range(ac_dim)], axis=0) for j in range(len(lp[0]))]
            # print "grad", grad

            if sm == []:
                sm = grad
                continue
            # print sess.run(sy_critic_g, feed_dict={sy_ob_no: ob_no, sy_ac_n: ac_n, sy_r_n: r_n, critic.sy_ac_n: ac_n, critic.sy_ob_no: ob_no})

            # sm = [grad1 + grad2 for grad1, grad2 in zip(sm, grad)]

            #sm = [grad / length for grad in sm]
            #fd = {i: d for i, d in zip(sy_g, sm)}
            # f1 = {sy_ob_no: ob_no, sy_ac_n: ac_n, sy_r_n: r_n, critic.sy_ac_n: ac_n, critic.sy_ob_no: ob_no, sy_tau:tau}
            '''
            # f1.update(fd)
            #sess.run(update_op, feed_dict={sy_ob_no: ob_no,
            #                                 critic.sy_ac_n: actor_mean,
            #                                 critic.sy_ob_no: ob_no})

            params = critic.GetParams()
            sess.run(critic_target.UpdateParams(params, tau))
            sess.run(actor_update, feed_dict={sy_tau: tau})






        # Log diagnostics
        logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
       # logz.log_tabular("KLOldNew", kl)
       # logz.log_tabular("Entropy", ent)
       # logz.log_tabular("EVBefore", explained_variance_1d(vpred_n, vtarg_n))
       # logz.log_tabular("EVAfter", explained_variance_1d(vf.predict(ob_no), vtarg_n))
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
        general_params = dict(gamma=0.97, animate=False, min_timesteps_per_batch=2500, n_iter=1000,
                              initial_stepsize=1e-7)
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

        main_pendulum(logdir=None, seed=0, desired_kl=2e-3, vf_type='nn', vf_params={}, gamma=0.99, animate=False,
                      min_timesteps_per_batch=1000, n_iter=1000, initial_stepsize=1e-6)
# import multiprocessing

#        p = multiprocessing.Pool()
#        p.map(main_pendulum1, params)
