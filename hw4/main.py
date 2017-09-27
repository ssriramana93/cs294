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
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
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

def gaussian_log_prob(mean, logstdev, ac_taken):
   

	dist = tf.contrib.distributions.MultivariateNormalDiag(loc = mean,scale_diag = tf.exp(logstdev))
	logprob = dist.log_prob(ac_taken)
	logprob = tf.Print(logprob, [logprob], message="This is LogProb: ")

	return logprob



def discount(x, gamma):
    """
    Compute discounted sum of future values
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y]. 
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1    
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

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
        A[np.arange(nfeats), np.arange(nfeats)] += 1e-3 # a little ridge regression
        b = Xp.T.dot(y)
        self.coef = np.linalg.solve(A, b)
    def predict(self, X):
        if self.coef is None:
            return np.zeros(X.shape[0])
        else:
            return self.preproc(X).dot(self.coef)
    def preproc(self, X):
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)

class NnValueFunction(object):
#	'''
	def __init__(self, ob_dim = 10, params = [32]):
		#self.graph = tf.Graph()
		self.alpha = 0
		#with self.graph.as_default():
		self.regloss = tf.zeros([1])
		self.sy_ob_nn = tf.placeholder(shape=[None, ob_dim], name="ob_nn", dtype=tf.float32)
		self.y = tf.placeholder(shape=[None, 1], name="y", dtype=tf.float32)

		#sy_h1 = lrelu(dense(sy_ob_nn, params[0], "sy_h1", weight_init=normc_initializer(1.0))) # hidden layer
		sy_h = tf.nn.elu(self.BN(self.dense(self.sy_ob_nn, params[0], "sh1", weight_init=tf.contrib.layers.xavier_initializer()),params[0]))
		#batch_mean2, batch_var2 = tf.nn.moments(z2_BN,[0])

		for i,l in enumerate(params[1:]):
			#tf.Variable("sy_h"+(i+2)) = lrelu(dense(tf.get_variable("sy_h" + (i+1)), params[0], "h"+l, weight_init=normc_initializer(1.0))) # hidden layer
			sy_h = tf.nn.elu(self.BN(self.dense(sy_h, l, "sh"+str(i + 2), weight_init=tf.contrib.layers.xavier_initializer()),l)) # hidden layer
			
		#self.sy_value = dense(tf.get_variable("sy_h" + len(params)),1, "op", weight_init=normc_initializer(1.0))
		self.sy_value = self.dense(sy_h, 1, "op", weight_init=tf.contrib.layers.xavier_initializer())
		self.loss = tf.reduce_mean(tf.square(self.sy_value - self.y)) + self.alpha*self.regloss
		self.sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32) # Symbolic, in case you want to change the stepsize during optimization. (We're not doing that currently)
		#self.optimizer = tf.train.AdamOptimizer(self.sy_stepsize).minimize(self.loss) 
		optimizer = tf.train.MomentumOptimizer(self.sy_stepsize, 0.7, use_nesterov = False)
		#optimizer = tf.train.AdamOptimizer(learning_rate=self.sy_stepsize)
		self.gvs = optimizer.compute_gradients(self.loss)
		capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in self.gvs]
		self.train_op = optimizer.apply_gradients(capped_gvs)

	

	def fit(self, X, y, nepoch = 100, init_step = 1e-2, minibatch_size = 64):
		#with self.graph.as_default():
		y = np.reshape(y, (len(y),1))
		index = np.arange(X.shape[0])
		ploss = 10000.0
		step = init_step
		tau = nepoch/10.0
		for e in xrange(nepoch):
			curr_index = np.random.choice(index, minibatch_size, replace=True)
			x_batch = X[curr_index, :]
			y_batch = y[curr_index, :]	
			loss, gvs = self.sess.run([self.loss, self.gvs], feed_dict = {self.sy_ob_nn: X, self.y:y})
			step = init_step*(tau/max(e, tau))**2
			print loss, step, np.sum([np.linalg.norm(grad) for grad, var in gvs])
			#if((loss[0] - ploss) > 0.1 and (e > 500)):
			#	step = step/(1.1)
			#ploss = loss	
			self.sess.run([self.train_op], feed_dict = {self.sy_ob_nn: x_batch, self.y:y_batch, self.sy_stepsize: step})
				
	def predict(self, X):
		value = self.sess.run([self.sy_value],feed_dict = {self.sy_ob_nn: X})
		#print "value", X.shape, np.shape(value[0])
		return np.squeeze(np.array(value))

	def registerSession(self, sess):
		self.sess = sess	

	def BN(self, layer, size):
		batch_mean2, batch_var2 = tf.nn.moments(layer,[0])
		scale2 = tf.Variable(tf.ones([size]))
		beta2 = tf.Variable(tf.zeros([size]))
		return tf.nn.batch_normalization(layer,batch_mean2,batch_var2,beta2,scale2,1e-8)	

    
	def dense(self, x, size, name, weight_init=None):
		"""
		Dense (fully connected) layer

		"""
		#with self.graph.as_default():
		w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
		b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
		self.regloss += tf.norm(w)
		return tf.matmul(x, w) + b	

	def lrelu(self, x, leak=0.2):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)
	
		


#	'''	

    #sy_h2 = lrelu(dense(sy_h1, 30, "h2", weight_init=normc_initializer(1.0)))
    #sy_h3 = lrelu(dense(sy_h2, 20, "h3", weight_init=normc_initializer(1.0)))
  #  sy_mean = dense(sy_ob_no, ac_dim, "mean", weight_init=normc_initializer(1.0))
 		#		for i,l in enumerate(layers):
#			sy_h1 = lrelu(dense() 
	pass # YOUR CODE HERE

def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)



def gaussian_sample_action(mean, logstdev, a_lb, a_ub):
#	mean = tf.Print(mean,[mean],"mean is:")
    
	U = mean + tf.exp(logstdev)*tf.random_normal(tf.shape(mean), mean = 0.0, stddev = 1.0)

	#U = tf.clip_by_value(U,a_lb,a_ub)
#	U = tf.Print(U,[U],"U is:")
	#lb_index = U < a_lb
	#ub_index = U > a_ub
#	print lb_index.shape, ub_index.shape

	#ub = lb = tf.ones(tf.shape(U))
	#lb = a_lb*lb
	#ub = a_ub*ub
	#lb = tf.reshape(np.ones()a_lb,tf.shape(U))
	#ub = tf.reshape(a_ub,tf.shape(U))
	#lb = a_lb
	#ub = a_ub
	#U = tf.where(lb_index,U,lb)
	#U = tf.where(ub_index,U,ub) 
	#U = tf.where(lb_index, tf.cast(lb,tf.float32), U)
	#U = tf.where(ub_index, tf.cast(ub,tf.float32), U)
	return U



def main_cartpole(n_iter=100, gamma=1.0, min_timesteps_per_batch=1000, stepsize=1e-2, animate=True, logdir=None):
    env = gym.make("CartPole-v0")
    ob_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    logz.configure_output_dir(logdir)
    vf = LinearValueFunction()

    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in these functionq
    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32) # batch of observations
    sy_ac_n = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) # batch of actions taken by the policy, used for policy gradient computation
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32) # advantage function estimate
    sy_h1 = lrelu(dense(sy_ob_no, 32, "h1", weight_init=normc_initializer(1.0))) # hidden layer
    sy_logits_na = dense(sy_h1, num_actions, "final", weight_init=normc_initializer(0.05)) # "logits", describing probability distribution of final layer
    # we use a small initialization for the last layer, so the initial policy has maximal entropy
    sy_oldlogits_na = tf.placeholder(shape=[None, num_actions], name='oldlogits', dtype=tf.float32) # logits BEFORE update (just used for KL diagnostic)
    sy_logp_na = tf.nn.log_softmax(sy_logits_na) # logprobability of actions
    sy_sampled_ac = categorical_sample_logits(sy_logits_na)[0] # sampled actions, used for defining the policy (NOT computing the policy gradient)
    sy_n = tf.shape(sy_ob_no)[0]
    sy_logprob_n = fancy_slice_2d(sy_logp_na, tf.range(sy_n), sy_ac_n) # log-prob of actions taken -- used for policy gradient calculation

    # The following quantities are just used for computing KL and entropy, JUST FOR DIAGNOSTIC PURPOSES >>>>
    sy_oldlogp_na = tf.nn.log_softmax(sy_oldlogits_na)
    sy_oldp_na = tf.exp(sy_oldlogp_na) 
    sy_kl = tf.reduce_sum(sy_oldp_na * (sy_oldlogp_na - sy_logp_na)) / tf.to_float(sy_n)
    sy_p_na = tf.exp(sy_logp_na)
    sy_ent = tf.reduce_sum( - sy_p_na * sy_logp_na) / tf.to_float(sy_n)
    # <<<<<<<<<<<<<

    sy_surr = - tf.reduce_mean(sy_adv_n * sy_logprob_n) # Loss function that we'll differentiate to get the policy gradient ("surr" is for "surrogate loss")

    sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32) # Symbolic, in case you want to change the stepsize during optimization. (We're not doing that currently)
    update_op = tf.train.AdamOptimizer(sy_stepsize).minimize(sy_surr)

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 
    # use single thread. on such a small problem, multithreading gives you a slowdown
    # this way, we can better use multiple cores for different experiments
    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101

    total_timesteps = 0

    for i in range(n_iter):
        print("********** Iteration %i ************"%i)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (i % 10 == 0) and animate)
            while True:
                if animate_this_episode:
                    env.render()
                obs.append(ob)
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})
                acs.append(ac)
                ob, rew, done, _ = env.step(np.clip(ac,-2,2))
                rewards.append(rew)
                if done:
                    break                    
            path = {"observation" : np.array(obs), "terminated" : terminated,
                    "reward" : np.array(rewards), "action" : np.array(acs)}
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
            adv_t = return_t - vpred_t
            advs.append(adv_t)
            vtargs.append(return_t)
            vpreds.append(vpred_t)

        # Build arrays for policy update
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_n = np.concatenate([path["action"] for path in paths])
        adv_n = np.concatenate(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        vtarg_n = np.concatenate(vtargs)
        vpred_n = np.concatenate(vpreds)
        vf.fit(ob_no, vtarg_n)

        # Policy update
        _, oldlogits_na = sess.run([update_op, sy_logits_na], feed_dict={sy_ob_no:ob_no, sy_ac_n:ac_n, sy_adv_n:standardized_adv_n, sy_stepsize:stepsize})
        kl, ent = sess.run([sy_kl, sy_ent], feed_dict={sy_ob_no:ob_no, sy_oldlogits_na:oldlogits_na})

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

def getGaussianKL(mean1, mean2, logstd1, logstd2, n):
	#d = shape[0]

	d = 1
	#std = tf.exp(logstd)
	std1 = tf.exp(logstd1)
	std2 = tf.exp(logstd2)

#	tr = (std1**2)/(std2**2)

	mean1 = tf.Print(mean1, [mean1, std1], message="This is Mean1: ")
	mean2 = tf.Print(mean2, [mean2, std2], message="This is Mean2: ")


	delMean = tf.cast(mean2 - mean1,tf.float32)
	delMean = tf.Print(delMean, [delMean], message="This is delMean: ")
	p = tf.log(std2/(std1 + 1e-8)) + ((std1**2) + (delMean)**2)/(2*(std2**2) + 1e-8)
	#p = tf.Print(p, [p, (std1**2 + (delMean)**2)/(2*(std2**2) + 1e-8), tf.log(std2/(std1 + 1e-8)) ], message = "This is p")
	return tf.reduce_mean(p - 0.5) 
#	return 0.5*tf.reduce_mean(tf.log((std2**2)/(std1**2)) + tr + tf.multiply(delMean,(delMean/(std2**2))) - d)


def getGaussianDiffEntropy(mean, logstd):

	std = tf.reduce_prod(tf.exp(logstd))
	diffEnt = 0.5*tf.log(2*np.pi*np.exp(1)*(std)**2)
	return diffEnt



def main_pendulum(logdir, seed, n_iter, gamma, min_timesteps_per_batch, initial_stepsize, desired_kl, vf_type, vf_params, animate=False):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env = gym.make("Pendulum-v0")
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    logz.configure_output_dir(logdir)
    if vf_type == 'linear':
        vf = LinearValueFunction(**vf_params)
    elif vf_type == 'nn':
        vf = NnValueFunction(ob_dim=ob_dim, **vf_params)

    print "bounds calculation"
    a_lb = env.action_space.low
    a_ub = env.action_space.high
    a_bnds = a_ub - a_lb

    print "Symbolic init"
    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32) # batch of observations
    sy_ac_n = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32) # batch of actions taken by the policy, used for policy gradient computation
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32) # advantage function estimate

    sy_h1 = lrelu(dense(sy_ob_no, 30, "h1", weight_init=tf.contrib.layers.xavier_initializer())) # hidden layer
    sy_h2 = lrelu(dense(sy_h1, 30, "h2", weight_init=tf.contrib.layers.xavier_initializer()))
    #sy_h3 = lrelu(dense(sy_h2, 32, "h3", weight_init=normc_initializer(1.0)))

    #sy_h3 = lrelu(dense(sy_h2, 20, "h3", weight_init=normc_initializer(1.0)))
    sy_mean = dense(sy_h2, ac_dim, "mean", weight_init=tf.contrib.layers.xavier_initializer())
    #sy_mean = tf.multiply(tf.cast(a_bnds,tf.float32),tf.tanh(dense(sy_h2, ac_dim, "mean", weight_init=normc_initializer(10.0))))

    sy_logstd = tf.get_variable("logstdev", [ac_dim], initializer=tf.zeros_initializer()) # Variance
#    sy_logstd = [tf.log(0.3)] 
    sy_sampled_ac = gaussian_sample_action(sy_mean,sy_logstd, a_lb, a_ub)

    sy_logprob_n = gaussian_log_prob(sy_mean,sy_logstd, sy_ac_n)
    sy_n = tf.shape(sy_ob_no)[0]

#    sy_logp_na = gaussian_log_prob(sy_mean,logstd_a,sy_sampled_ac)
    #oldlogstd_a = 
    sy_oldmean = tf.placeholder(shape=[None, ac_dim], name='oldmean', dtype=tf.float32)
    sy_oldlogstd = tf.placeholder(shape=[None, ac_dim], name='oldlogstdev', dtype=tf.float32)

    #sy_oldlogp_na = tf.placeholder(shape=[None, ac_dim], name='oldlogprob', dtype=tf.float32)

    sy_kl = getGaussianKL(sy_oldmean, sy_mean, sy_oldlogstd, sy_logstd, sy_n)

   
 
    sy_ent = getGaussianDiffEntropy(sy_mean, sy_logstd)



    

    print "D1"

    sy_surr = - tf.reduce_mean(tf.multiply(sy_adv_n,sy_logprob_n)) # Loss function that we'll differentiate to get the policy gradient ("surr" is for "surrogate loss")
    sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32) # Symbolic, in case you want to change the stepsize during optimization. (We're not doing that currently)
    update_op = tf.train.AdamOptimizer(sy_stepsize).minimize(sy_surr)

    sess = tf.Session()
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101
    vf.registerSession(sess)
    print "D2"
    #testShit>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    s_ac = []
    sy_testmean = tf.placeholder(shape=[None, ac_dim], name='oldmean', dtype=tf.float32)
    sy_testlogstd = tf.placeholder(shape=[None, ac_dim], name='oldlogstdev', dtype=tf.float32)
    sy_test_a = tf.placeholder(shape=[None, ac_dim], name='oldlogstdev', dtype=tf.float32)

    sy_sampled_ac_test = gaussian_sample_action(sy_testmean, sy_testlogstd, a_lb, a_ub)
    sy_logprob_test = gaussian_log_prob(sy_testmean,sy_testlogstd, sy_test_a)
    mean_t = np.reshape(np.array([[0.113],[0.111]]),(2,1))
    std_t = np.reshape(np.array([np.log(0.001)]),(1,1))
    print a_lb, a_ub

    for i in xrange(1000):
      a = sess.run([sy_sampled_ac_test], feed_dict={sy_testmean:mean_t,sy_testlogstd:std_t})
      s_ac.append(a)      
    s_ac = np.array(s_ac)

    print s_ac   
    print "std = ", np.std(s_ac, axis = 0),"mean= ", np.mean(s_ac, axis = 0)
    lp = sess.run([sy_logprob_test], feed_dict={sy_testmean:mean_t,sy_testlogstd:std_t,sy_test_a:np.reshape(s_ac[0],(2,1))})
    from scipy.stats import norm
    #print s_ac[0][0], s_ac[1][0]
    logprob = np.array([norm.logpdf(s_ac[0][0][0], loc = mean_t[0,:], scale = np.exp(std_t)), norm.logpdf(s_ac[0][0][1], loc = mean_t[1,:], scale = np.exp(std_t))])
    #print "logprob",logprob,"s_ac",s_ac[0][0],"s_ac",s_ac[1][0]
    print "lp_score", np.sum(lp - logprob)
    #testShit>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    total_timesteps = 0
    stepsize = initial_stepsize
    import time

    for i in range(n_iter):
        print("********** Iteration %i ************"%i)

        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (i % 10 == 0) and animate)
            while True:
                if animate_this_episode:
                    env.render()
                ob = ob.reshape((1,ob_dim))    
    
                obs.append(ob)
#                ob = ob.reshape((1,ob_dim))    

                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : np.reshape(ob,(1,ob_dim))})
                acs.append(ac)
                
                ob, rew, done, _ = env.step(np.clip(ac,-2.0,2.0))
                rewards.append(rew)
                if done:
                    break
        #    print "path", len(obs), np.array(obs)
            path = {"observation" : np.squeeze(np.array(obs)), "terminated" : terminated,
                    "reward" : np.squeeze(np.array(rewards)), "action" : np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
           # print timesteps_this_batch
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
        ac_n = np.reshape(ac_n,(ac_n.shape[0],ac_dim))
        vf.fit(ob_no, vtarg_n)
        #sh=sy_mean.get_shape().as_list()
        #print sh
        #oldoldlogstd = 1.01
        

#        sy_surr = tf.Print(sy_surr,[sy_surr],"Loss =")  
        print "Loss = ", sess.run(sy_surr,feed_dict={sy_ob_no:ob_no, sy_ac_n:ac_n, sy_adv_n:standardized_adv_n, sy_stepsize:stepsize})  
        # Policy update
        oldmean,oldlogstd,oldlogprob = sess.run([sy_mean,sy_logstd,sy_logprob_n], feed_dict={sy_ob_no:ob_no, sy_ac_n:ac_n, sy_adv_n:standardized_adv_n, sy_stepsize:stepsize})
        sess.run([update_op], feed_dict={sy_ob_no:ob_no, sy_ac_n:ac_n, sy_adv_n:standardized_adv_n, sy_stepsize:stepsize})

        #print "ac = ", ac_n, "oldmean = ", oldmean,"oldlogstd = ", oldlogstd
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        '''std1 = np.exp(oldoldlogstd)
        std2 = np.exp(oldlogstd)
        tr = (std1**2)/(std2**2)
        delMean = oldoldmean - oldmean
        print 0.5*np.mean(np.log((std2**2)/(std1**2)) + tr + np.multiply(delMean,delMean/(std2**2)) - 1)
        oldoldmean = oldmean
        oldoldlogstd = oldlogstd'''
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        kl, ent = sess.run([sy_kl, sy_ent], feed_dict={sy_ob_no:ob_no, sy_oldmean:oldmean, sy_oldlogstd: np.reshape(oldlogstd,[1,ac_dim])})
        #oldoldmean = oldmean
       # if (n_iter > 150):
       # 	desired_kl = 1e-5
        	
        print "Done"
        if kl > desired_kl * 2: 
            stepsize /= 1.5
            print('stepsize -> %s'%stepsize)
        elif kl < desired_kl / 2: 
            stepsize *= 1.5
            print('stepsize -> %s'%stepsize)
        else:
            print('stepsize OK')

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
        main_cartpole(logdir=None) # when you want to start collecting results, set the logdir

    if 1:
        general_params = dict(gamma=0.97, animate=False, min_timesteps_per_batch=2500, n_iter=1000, initial_stepsize=1e-7)
        params = [

            dict(logdir= None , seed=0, desired_kl=2e-6, vf_type='linear', vf_params={}, **general_params),

 #           dict(logdir=  '/tmp/ref11/linearvf-kl2e-3-seed0' , seed=0, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
#            dict(logdir=  '/tmp/ref0/nnvf-kl2e-3-seed0' , seed=0, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
#            dict(logdir=  '/tmp/ref0/linearvf-kl2e-3-seed1' , seed=1, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
#            dict(logdir=  '/tmp/ref0/nnvf-kl2e-3-seed1' , seed=1, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
#            dict(logdir=  '/tmp/ref0/linearvf-kl2e-3-seed2' , seed=2, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
#            dict(logdir=  '/tmp/ref0/nnvf-kl2e-3-seed2' , seed=2, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
        ]
        p = dict(logdir= None , seed=0, desired_kl=2e-6, vf_type='linear', vf_params={}, **general_params),

        main_pendulum(logdir=None , seed=0, desired_kl=2e-3, vf_type='nn', vf_params={}, gamma=0.97, animate=False, min_timesteps_per_batch=2500, n_iter=1000, initial_stepsize=1e-6)
#        import multiprocessing

#        p = multiprocessing.Pool()
#        p.map(main_pendulum1, params)
