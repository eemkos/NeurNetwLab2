import numpy as np

"""
def dot(op1, op2):
    assert hasattr(op1, 'shape') and hasattr(op2, 'shape')
    #print(op1.shape, op2.shape)
    assert op1.shape[1] == op2.shape[0]

    result = np.zeros(shape=(op1.shape[0], op2.shape[1]))

    for i in range(op1.shape[0]):
        for j in range(op2.shape[1]):
            result[i, j] = sum(map(lambda xy: xy[0] * xy[1], zip(op1[i, :], op2[:, j])))

    return result


def transpose(matrix):
    assert hasattr(matrix, 'shape')

    result = np.zeros(shape=(matrix.shape[1], matrix.shape[0]))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            result[j, i] = matrix[i, j]

    return result


def add(op1, op2):
    assert hasattr(op1, 'shape')
    assert hasattr(op2, 'shape') or np.isscalar(op2)
    result = np.zeros(shape=op1.shape)

    if np.isscalar(op2):
        for i in range(op1.shape[0]):
            for j in range(op1.shape[1]):
                result[i, j] = op1[i, j] + op2
    else:
        assert op1.shape == op2.shape
        for i in range(op1.shape[0]):
            for j in range(op1.shape[1]):
                result[i, j] = op1[i, j] + op2[i, j]

    return result


def subtract(op1, op2):
    assert hasattr(op1, 'shape') or hasattr(op2, 'shape')

    if np.isscalar(op2):
        result = np.zeros(shape=op1.shape)
        for i in range(op1.shape[0]):
            for j in range(op1.shape[1]):
                result[i, j] = op1[i, j] - op2
    elif np.isscalar(op1):
        result = np.zeros(shape=op2.shape)
        for i in range(op2.shape[0]):
            for j in range(op2.shape[1]):
                result[i, j] = op1 - op2[i,j]
    else:
        #print(op1.shape, op2.shape)
        assert op1.shape == op2.shape
        result = np.zeros(shape=op1.shape)
        for i in range(op1.shape[0]):
            for j in range(op1.shape[1]):
                result[i, j] = op1[i, j] - op2[i, j]

    return result


def mult(op1, op2):
    assert hasattr(op1, 'shape') or (np.isscalar(op1) and hasattr(op2, 'shape'))
    assert hasattr(op2, 'shape') or (np.isscalar(op2) and hasattr(op1, 'shape'))

    if np.isscalar(op2):
        result = np.zeros(shape=op1.shape)
        for i in range(op1.shape[0]):
            for j in range(op1.shape[1]):
                result[i, j] = op1[i, j] * op2
    elif np.isscalar(op1):
        result = np.zeros(shape=op2.shape)
        for i in range(op2.shape[0]):
            for j in range(op2.shape[1]):
                result[i, j] = op2[i, j] * op1
    else:
        assert op1.shape == op2.shape
        result = np.zeros(shape=op1.shape)
        for i in range(op1.shape[0]):
            for j in range(op1.shape[1]):
                result[i, j] = op1[i, j] * op2[i, j]

    return result


def div(op1, op2):
    assert hasattr(op1, 'shape') or hasattr(op2, 'shape')


    if np.isscalar(op2):
        result = np.zeros(shape=op1.shape)
        for i in range(op1.shape[0]):
            for j in range(op1.shape[1]):
                result[i, j] = op1[i, j] / op2
    elif np.isscalar(op1):
        result = np.zeros(shape=op2.shape)
        for i in range(op2.shape[0]):
            for j in range(op2.shape[1]):
                result[i, j] = op1 / op2[i, j]
    else:
        assert op1.shape == op2.shape
        result = np.zeros(shape=op1.shape)
        for i in range(op1.shape[0]):
            for j in range(op1.shape[1]):
                result[i, j] = op1[i, j] / op2[i, j]

    return result


def are_equal(op1, op2):
    assert hasattr(op1, 'shape') and hasattr(op2, 'shape')
    assert op1.shape == op2.shape

    for row1, row2 in zip(op1, op2):
        if not all(x1 == x2 for x1, x2 in zip(row1, row2)):
            return False
    return True


def repeat_row(row_vector, nb_rows):
    return np.asarray([row_vector for _ in range(nb_rows)])


def exp(matrix):
    if np.isscalar(matrix):
        return np.exp(matrix)
    else:
        result = np.zeros(matrix.shape)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                result[i, j] = np.exp(matrix[i, j])

        return result


def neg(matrix):
    if np.isscalar(matrix):
        return -matrix
    else:
        result = np.zeros(matrix.shape)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                result[i, j] = -matrix[i, j]

        return result


def sum_rows(matrix):
    if np.isscalar(matrix):
        return matrix
    else:
        result = np.zeros((1, matrix.shape[1]))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                result[1, j] += matrix[i, j]

        return result


def calc_sum(matrix, axis=None):
    '''if axis is None:
        res = 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                res += matrix[i,j]
    elif axis == 0:
        res = np.zeros((matrix.shape[0], 1))
        '''
    return np.sum(matrix, axis=axis)


def ln(matrix):
    res = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            res[i, j] = np.log(matrix[i, j])
    return res

"""

def dot(op1, op2):
    return op1.dot(op2)


def transpose(matrix):
    return matrix.T


def add(op1, op2):
    return op1 + op2


def subtract(op1, op2):
    return op1 - op2


def mult(op1, op2):
    return op1 * op2


def div(op1, op2):
    return op1/op2


def are_equal(op1, op2):
    assert hasattr(op1, 'shape') and hasattr(op2, 'shape')
    assert op1.shape == op2.shape

    for row1, row2 in zip(op1, op2):
        if not all(x1 == x2 for x1, x2 in zip(row1, row2)):
            return False
    return True


def repeat_row(row_vector, nb_rows):
    return np.asarray([row_vector for _ in range(nb_rows)])


def exp(matrix):
    return np.exp(matrix)


def neg(matrix):
    return -matrix


def sum_rows(matrix):
    return np.sum(matrix, axis=1)


def calc_sum(matrix, axis=None):
    return np.sum(matrix, axis=axis)


def ln(matrix):
    eps = 0.00001
    return np.log(matrix+eps)

