import numpy as np
import tensorflow as tf
import copy

def findDiff(x_list, y_list):
    return [x_list[i] - y_list[i] for i, val in enumerate(x_list)]


def findSum2(x_list, y_list):
    return [x_list[i] + y_list[i] for i, val in enumerate(x_list)]


def findProd(x_list, y_list):
    return [tf.multiply(x_list[i], y_list[i]) for i, val in enumerate(x_list)]


def findSum(x_list):
    return tf.reduce_sum([tf.reduce_sum(x_list[i]) for i, val in enumerate(x_list)])


def findMul(x_list, scalar):
    return [scalar * val for val in x_list]


def findSumAlongAxis(x_list):
    if x_list == []:
        return []
    sum = x_list[0]
    for i, val in enumerate(x_list):
        sum = findSum2(sum, val)
    return sum


def dotList(x, y):
    return tf.reduce_sum([tf.reduce_sum(x[i] * y[i]) for i in xrange(len(x))])



def printList(x_list, strg='No'):
    return [tf.Print(x_list[i], [x_list[i]], strg) for i, val in enumerate(x_list)]


def removeNan(x_list):
    return [tf.where(tf.is_nan(x_list[i]), tf.zeros_like(x_list[i]), x_list[i]) for i, val in enumerate(x_list)]



def flatten1(varlist):
    flatvar = tf.concat([tf.reshape(var, [-1]) for var in varlist], axis = 0)
    return flatvar


def flatten(gradvarlist):
    grad, var = zip(*gradvarlist)
    flatgrad = flatten1(grad)
    flatvar = flatten1(var)
    return flatgrad, flatvar




def unflatten(flatgrad, flatvar, gradvarlist):
    index = [0, 0]
    gradvarlist1 = []
    for i, gradvar in enumerate(gradvarlist):
        grad, var = gradvar
        shape = tf.shape(var)
        # grad = tf.reshape(tf.slice(flatgrad,index,[0, tf.size(grad)], shape))
        sliced_var = tf.slice(flatvar, index, [1, tf.size(var)])
        sliced_grad = tf.slice(flatgrad, index, [1, tf.size(var)])
        print gradvarlist[i][0], tf.__version__
        gradvarlist1.append((tf.reshape(sliced_var, shape), tf.reshape(sliced_grad, shape)))
        #  print sliced_var.get_shape().as_list()
        # gradvarlist[i][0] = tf.assign(ref = gradvarlist[i][0], value = tf.reshape(sliced_var, shape), validate_shape = True)
        # gradvarlist[i][1] = tf.assign(ref = gradvarlist[i][1], value = tf.reshape(sliced_grad, shape), validate_shape = True)

        index = [0, index[1] + tf.size(var)]
        print gradvarlist[i]

    return gradvarlist1


def unflatten1(flatvar, varlist):
    index = [0, 0]
    for i, var in enumerate(varlist):
        shape = tf.shape(var)
        # grad = tf.reshape(tf.slice(flatgrad,index,[0, tf.size(grad)], shape))
        sliced_var = tf.slice(flatvar, index, [1, tf.size(var)])
        #  print sliced_var.get_shape().as_list()
        varlist[i] = tf.assign(varlist[i], tf.reshape(sliced_var, shape))
        index = [0, index[1] + tf.size(var)]

    return varlist





def dotProdList(x_list, y_list):
    #return np.dot(x_list[0].T, y_list[0])
    return (np.sum([np.sum(np.multiply(np.array(x), np.array(y))) for x,y in zip(x_list, y_list)]))

def scalarProdList(x_list, scalar):
   return [scalar * val for val in x_list]

def Add2List(x_list, y_list):
    #return [x_list[0] + y_list[0]]
    return [x_list[i] + y_list[i] for i, val in enumerate(x_list)]

def Sub2List(x_list, y_list):
    #return [x_list[0] - y_list[0]]
    return [x_list[i] - y_list[i] for i, val in enumerate(x_list)]

def Mul2List(x_list, y_list):
    #return [x_list[0] - y_list[0]]
    return [np.multiply(x_list[i],y_list[i]) for i, val in enumerate(x_list)]

def SumAList(x_list):
    return np.sum([np.sum(val) for val in x_list])


def CG(HVP, y, f_dict, cgIter=10,eps = 1e-8):
    x = [np.random.rand(*th.shape) for th in y]

    r = Sub2List(copy.deepcopy(y),HVP(x,f_dict))
   # print 'r', r
    p = copy.deepcopy(r)
    rolddot = dotProdList(r, r)

    for k in xrange(cgIter):


        hvp_p = HVP(p,f_dict)
        alpha = rolddot/(dotProdList(p,hvp_p) + eps)
        x = Add2List(x,scalarProdList(p,alpha))
        r = Sub2List(r,scalarProdList(hvp_p,alpha))
        rnewdot = dotProdList(r,r)

        beta = rnewdot/(rolddot + eps)
        if(np.sqrt(beta) < eps):
            break
        p = Add2List(r, scalarProdList(p, beta))
        rolddot = rnewdot
    #print "x = ", x
    #x = [np.clip(val, -1., 1.) for val in x]
    return x
