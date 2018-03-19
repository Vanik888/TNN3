#!/usr/bin/env python2.7

import numpy as np
import os
from argparse import ArgumentParser

parser = ArgumentParser(description='assignment3')
parser.add_argument('--show-gnuplot-cmd',
                    action='store_true',
                    dest='show_gnuplot_cmd',
                    default=False,
                    help='Show run command for gnuplot'
                    )

parser.add_argument('--test-steps',
                    type=int,
                    default=10001,
                    dest='test_steps',
                    help='Number of steps for learning process',
                    )

result_file = 'learning.curve'
training_file = 'training.dat'
test_file = 'test.dat'
max_weight = 2
min_weight = -2
current_dir = os.path.dirname(os.path.abspath(__file__))
result_file_full_path = '/'.join([current_dir, result_file])


def get_matrixes_from_file(training_file):
    """
    :param training_file: file with training data
    :return: list with X and Y
    """
    lines = []
    with open(training_file, 'r') as f:
        lines = f.readlines()
        lines = [l for l in lines if not l.startswith('#') and not
        l.startswith(' ')]
    split_var = '    ' if '    ' in lines[0] else '  '
    X = [[float(k) for k in l.split(split_var)[:1][0].split()] for l in lines]
    Y = [[float(k) for k in l.split(split_var)[1:][0].split()] for l in lines]
    return add_bayes_column(np.array(X)), np.array(Y)


def generate_weights(size, min_border=min_weight, max_border=max_weight):
    """
    :param size: matrix size
    :param min_border:  min weight value
    :param max_border: max weight value
    :return: matrix of weights
    """
    return np.random.uniform(min_border, max_border, size)


def print_weights(weights):
    """
    :param weights: matrix of weights
    :return: None
    """
    for i, w in enumerate(weights):
        print('w%s=\n%s\n' % (i, w))


def add_bayes_column(z):
    """
    :param z: initial matrix without bias column
    :return: matrix z with bias as a first column
    """
    m = np.ones((z.shape[0], z.shape[1]+1))
    m[:, 1:] = z
    return m


def transfer_fn(x, fn='logistic', derivative=False):
    """
    :param x: matrix
    :param fn: type of returned transfer function
    :param derivative: return a derivative
    :return: transfer function results
    """
    if fn == 'logistic':
        return x*(1-x) if derivative else 1/(1+np.exp(-x))
    if fn == 'tanh':
        return 1 - x*x if derivative else (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))


# number of layers in network
layers = 4
# number of neurons in hidden layer
h = 3
# number of neurons in the next hidden layer
k = 2
# X = PxN; Y=PxM
X, Y = get_matrixes_from_file(training_file)
# number of outputs
m = Y.shape[1]
# number of examples
p = X.shape[0]

np.random.seed(100)
# N+1xH+1
w1 = generate_weights((X.shape[1], h+1))
# H+1xK+1
w2 = generate_weights((h+1, k+1))
# K+1xM
w3 = generate_weights((k+1, Y.shape[1]))

eta1 = 2.5
eta2 = 2.6
eta3 = 2.7

#TODO: add identity transfer funcion
# add readme file
def make_test(w1, w2, w3, input_file=test_file):
    print('Test trained network.')
    X, Y = get_matrixes_from_file(input_file)
    print('Input X=\n%s' % X)
    print('Output Y=\n%s' % Y)
    # Px(H+1) = (PxN+1)*(N+1xH+1)
    out1 = transfer_fn(np.dot(X, w1), fn='tanh')
    # Px(K+1) = (Px(H+1))*((H+1)x(K+1))
    out2 = transfer_fn(np.dot(out1, w2), fn='logistic')
    # PxM = (Px(K+1))*((K+1)xM)
    out3 = transfer_fn(np.dot(out2, w3), fn='logistic')
    error = Y - out3
    print('Error=\n%s' % error.T)


def calc(w1, w2, w3, test_steps, show_gnuplot_cmd=False):
    # clean results file
    with open(result_file, 'w') as f:
        f.write('')
    for i in xrange(test_steps):
        # Px(H+1) = (PxN+1)*(N+1xH+1)
        out1 = transfer_fn(np.dot(X, w1), fn='tanh')
        # Px(K+1) = (Px(H+1))*((H+1)x(K+1))
        out2 = transfer_fn(np.dot(out1, w2), fn='logistic')
        # PxM = (Px(K+1))*((K+1)xM)
        out3 = transfer_fn(np.dot(out2, w3), fn='logistic')
        error = Y - out3
        # PxM
        delta3 = error*transfer_fn(out3, fn='logistic', derivative=True)
        # Px(K+1) = PxM * Mx(K+1)
        delta2 = np.dot(delta3, w3.T)*transfer_fn(out2,
                                                  fn='logistic',
                                                  derivative=True)
        # Px(H+1) = Px(K+1) * (K+1)x(H+1)
        delta1 = np.dot(delta2, w2.T) * transfer_fn(out1,
                                                    fn='tanh',
                                                    derivative=True)
        # (N+1)x(H+1) = (N+1)xP * Px(H+1)
        w1 += eta1*np.dot(X.T, delta1)
        # (H+1)x(K+1) = (H+1)xP * Px(K+1)
        w2 += eta2*np.dot(out1.T, delta2)
        # (K+1)xM = (K+1)xP * PxM
        w3 += eta3*np.dot(out2.T, delta3)
        with open(result_file, 'a') as f:
            common_error_str = ''
            for j in xrange(m):
                common_error_str += ' '.join([str(e) for e in error.T[j]])
            f.write("%s %s\n" % (i, common_error_str))
    print('gnuplot command to draw errors:')
    if show_gnuplot_cmd:
        plotter_str = '"%s" using 1:%s title \'M=%s; P=%s\', \\\n'
        plotter_cmd_list = []
        for i in xrange(m):
            plotter_cmd_list.append('plot ')
            for k in xrange(2+i*p, p*(i+1)+2):
                # from second column till p+2
                plotter_cmd_list.append(plotter_str % (result_file_full_path,
                                                       k, i+1, k-1-i*p))

        print(''.join(plotter_cmd_list))
    return w1, w2, w3


if __name__ == '__main__':
    args = parser.parse_args()
    test_steps = args.test_steps
    show_gnuplot_cmd = args.show_gnuplot_cmd
    print('network=N-H-K-M')
    print('N=%s\nH=%s\nK=%s\nM=%s' % (X.shape[1]-1, h, k, m))
    for i, eta in enumerate([eta1, eta2, eta3]):
        print('eta%s=%s' % (i, eta))
    print('Initial weights: ')
    print_weights([w1, w2, w3])
    w1, w2, w3 = calc(w1, w2, w3, test_steps, show_gnuplot_cmd)
    print('Trained weights: ')
    print_weights([w1, w2, w3])
    make_test(w1, w2, w3, test_file)

