from numpy import random, dot, array, ones, exp

training_file = 'training0.dat'
max_weight = 2
min_weight = 2


# 19:30
def get_training_data(training_file):
    """
    :param training_file: file with training data
    :return: list with X and Y
    """
    lines = []
    with open(training_file, 'r') as f:
        lines = f.readlines()
        lines = [l for l in lines if not l.startswith('#')]
    X = [[float(k) for k in l.split('  ')[:1][0].split()] for l in lines]
    Y = [[float(k) for k in l.split('  ')[1:][0].split()] for l in lines]
    return array(X), array(Y)


def generate_weights(size, min_border=min_weight, max_border=max_weight):
    return random.uniform(min_border, max_border, size)


def add_bayes_column(z):
    """
    :param z: initial matrix without bias column
    :return: matrix z with bias as a first column
    """
    m = ones((z.shape[0], z.shape[1]+1))
    m[:, 1:] = z
    return m


def transfer_fn(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+exp(-x))


# number of layers in network
layers = 4
# number of neurons in hidden layer
h = 3
# number of neurons in the next hidden layer
k = 2
# X = PxN; Y=PxM
X, Y = get_training_data(training_file)
X = add_bayes_column(X)

random.seed(100)
# N+1xH+1
w1 = generate_weights((X.shape[1], h+1))
# H+1xK+1
w2 = generate_weights((h+1, k+1))
# K+1xM
w3 = generate_weights((k+1, Y.shape[1]))

eta1 = 2.5
eta2 = 2.6
eta3 = 2.7


def calc(w1, w2, w3):
    for i in xrange(100001):
        # Px(H+1) = (PxN+1)*(N+1xH+1)
        out1 = transfer_fn(dot(X, w1))
        # Px(K+1) = (Px(H+1))*((H+1)x(K+1))
        out2 = transfer_fn(dot(out1, w2))
        # PxM = (Px(K+1))*((K+1)xM)
        out3 = transfer_fn(dot(out2, w3))
        error = Y - out3
        # PxM
        delta3 = error*transfer_fn(out3, derivative=True)
        # Px(K+1) = PxM * Mx(K+1)
        delta2 = dot(delta3, w3.T)*transfer_fn(out2, derivative=True)
        # Px(H+1) = Px(K+1) * (K+1)x(H+1)
        delta1 = dot(delta2, w2.T)*transfer_fn(out1, derivative=True)
        # (N+1)x(H+1) = (N+1)xP * Px(H+1)
        w1 += eta1*dot(X.T, delta1)
        # (H+1)x(K+1) = (H+1)xP * Px(K+1)
        w2 += eta2*dot(out1.T, delta2)
        # (K+1)xM = (K+1)xP * PxM
        w3 += eta3*dot(out2.T, delta3)
        if i % 1000 == 0:
            print(error.T)


if __name__ == '__main__':
    calc(w1, w2, w3)





