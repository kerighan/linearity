# distutils: language = c++

#!python
#cython: language_level=3
#cython.wraparound=False
#cython.boundscheck=False
#cython.nonecheck=False
from cpython cimport array
from cython.view cimport array as cvarray
cimport cython
from libcpp.algorithm cimport sort
from libc.stdlib cimport malloc, free
from itertools import chain
import array


cdef class vector(object):
    cdef public array.array value
    cdef public float[:] v
    cdef public Py_ssize_t size

    def __init__(self, list data):
        self.value = array.array('f', data)
        self.v = self.value
        self.size = len(self.value)

    def __repr__(self):
        return str(list(self.value))

    def __getitem__(self, int key):
        return self.v[key]

    def __add__(self, v):
        if isinstance(v, vector):
            return self.add(v)
        else:
            return self.add_scalar(v)

    def __sub__(self, v):
        if isinstance(v, vector):
            return self.sub(v)
        else:
            return self.add_scalar(-v)

    def __rmul__(self, v):
        return self.__mul__(v)

    def __mul__(self, v):
        if isinstance(v, vector):
            return self.multiply(v)
        else:
            return self.multiply_scalar(v)

    def __truediv__(self, v):
        if isinstance(v, vector):
            return self.divide(v)
        else:
            return self.multiply_scalar(1 / v)

    def __list__(self):
        return list(self.value)

    def __pow__(self, e, v):
        return self.pow(e)

    def __ge__(self, hyperplane h):
        return h.side_of(self)

    def __setitem__(self, int key, float value):
        self.v[key] = value

    def __eq__(self, vector v):
        for i in range(self.size):
            if v.v[i] != self.v[i]:
                return False
        return True

    # stats
    cpdef sum(self):
        cdef float sum_ = 0
        for i in range(self.size):
            sum_ += self.v[i]
        return sum_

    cpdef mean(self):
        cpdef float sum_ = self.sum()
        return sum_ / self.size
    
    cpdef var(self):
        cpdef float var_ = 0
        cpdef float mean_ = self.mean()
        for i in range(self.size):
            var_ += (self.v[i] - mean_)**2
        return var_ / self.size

    cpdef median(self):
        cdef int size = self.size
        cdef int index = size // 2
        if size % 2 == 1:
            index += 1
        cdef array.array value = array.copy(self.value)
        
        sort_cpp(value)
        return value[index]

    def is_null(self):
        return self.norm() == 0

    # algebra
    cpdef vector add(self, vector v):
        cdef array.array value = array.copy(self.value)
        cdef float[:] view = value
        cdef vector new_vector = vector([])

        for i in range(self.size):
            view[i] += v.v[i]

        new_vector.value = value
        new_vector.v = view
        new_vector.size = self.size
        return new_vector

    cpdef vector sub(self, vector v):
        cdef array.array value = array.copy(self.value)
        cdef float[:] view = value
        cdef vector new_vector = vector([])

        for i in range(self.size):
            view[i] -= v.v[i]

        new_vector.value = value
        new_vector.v = view
        new_vector.size = self.size
        return new_vector

    cpdef vector add_scalar(self, float v):
        cdef array.array value = array.copy(self.value)
        cdef float[:] view = value
        cdef vector new_vector = vector([])

        for i in range(self.size):
            view[i] += v

        new_vector.value = value
        new_vector.v = view
        new_vector.size = self.size
        return new_vector

    cpdef vector multiply(self, vector v):
        cdef array.array value = array.copy(self.value)
        cdef float[:] view = value
        cdef vector new_vector = vector([])

        for i in range(self.size):
            view[i] *= v.v[i]

        new_vector.value = value
        new_vector.v = view
        new_vector.size = self.size
        return new_vector

    cpdef vector multiply_scalar(self, float v):
        cdef array.array value = array.copy(self.value)
        cdef float[:] view = value
        cdef vector new_vector = vector([])

        for i in range(self.size):
            view[i] *= v

        new_vector.value = value
        new_vector.v = view
        new_vector.size = self.size
        return new_vector

    cpdef vector divide(self, vector v):
        cdef array.array value = array.copy(self.value)
        cdef float[:] view = value
        cdef vector new_vector = vector([])

        for i in range(self.size):
            view[i] /= v.v[i]

        new_vector.value = value
        new_vector.v = view
        new_vector.size = self.size
        return new_vector
    
    cpdef vector pow(self, float v):
        cdef array.array value = array.copy(self.value)
        cdef float[:] view = value
        cdef vector new_vector = vector([])

        for i in range(self.size):
            view[i] = view[i]**v

        new_vector.value = value
        new_vector.v = view
        new_vector.size = self.size
        return new_vector

    # utils
    cpdef int argmax(self):
        if self.size == 0:
            return 0

        cdef float max_ = self.value[0]
        cdef int index_max = 0
        cdef Py_ssize_t i
        cdef float value
        for i in range(1, self.size):
            value = self.v[i]
            if value > max_:
                index_max = i
                max_ = value
        return index_max

    cpdef float max(self):
        if self.size == 0:
            return 0

        cdef float max_ = self.v[0]
        cdef Py_ssize_t i
        cdef float value
        for i in range(1, self.size):
            value = self.v[i]
            if value > max_:
                max_ = value
        return max_

    cpdef float min(self):
        if self.size == 0:
            return 0

        cdef float min_ = self.v[0]
        cdef Py_ssize_t i
        cdef float value
        for i in range(1, self.size):
            value = self.v[i]
            if value < min_:
                min_ = value
        return min_
    
    # vector operation
    cpdef float norm(self):
        cdef float norm_ = 0
        for i in range(self.size):
            norm_ += self.v[i]**2
        return norm_**.5

    def normalize(self):
        cdef array.array value = array.copy(self.value)
        cdef float[:] view = value
        cdef vector new_vector = vector([])
        cdef float norm_ = self.norm()

        for i in range(self.size):
            view[i] /= norm_

        new_vector.value = value
        new_vector.v = view
        new_vector.size = self.size
        return new_vector
    
    cpdef distance_from(self, item):
        if isinstance(item, vector):
            return (self - item).norm()
        elif isinstance(item, hyperplane):
            return dot(self - item.position, item.normal)


cdef class matrix(object):
    cdef public array.array value
    cdef public float[:] v
    cdef public (Py_ssize_t, Py_ssize_t) shape
    cdef float *my_array

    def __init__(self, list data):
        cdef Py_ssize_t rows = len(data)
        cdef Py_ssize_t cols = len(data[0])
        self.shape = (rows, cols)
        cdef Py_ssize_t i
        cdef Py_ssize_t j
        my_array = cvarray(shape=(rows, cols), itemsize=sizeof(float), format="f")

        for i in range(rows):
            for j in range(cols):
                my_array[i, j] = data[i][j]


cdef class hyperplane(object):
    cdef public vector normal
    cdef public vector position

    def __init__(self, vector normal, vector position=None, int to_norm=True):
        if to_norm:
            self.normal = normal.normalize()
        else:
            self.normal = normal
        if position is None:
            self.position = zeros(normal.size)
        else:
            self.position = position

    cpdef int side_of(self, vector v):
        cpdef dot_ = 0
        for i in range(v.size):
            dot_ += (v.v[i] - self.position.v[i]) * self.normal.v[i]
        if dot_ >= 0:
            return 1
        return 0

    cpdef list to_list(self):
        return [list(self.normal.value), list(self.position.value)]


# preset
cpdef zeros(dim):
    return vector([0] * dim)


cpdef ones(dim):
    return vector([1] * dim)


cpdef random(dim):
    import random as rd
    return vector([rd.random()] * dim)


# vector operations
cpdef float dot(vector a, vector b):
    cdef float dot_value = 0
    for i in range(a.size):
        dot_value += a.v[i] * b.v[i]
    return dot_value


cpdef vector center(vector a, vector b):
    cdef array.array value = array.copy(a.value)
    cdef float[:] view = value
    cdef vector new_vector = vector([])

    for i in range(a.size):
        view[i] = (view[i] + b.v[i]) / 2

    new_vector.value = value
    new_vector.v = view
    new_vector.size = a.size
    return new_vector


cpdef vector vector_from_axis(list X, int dim):
    cdef list value = []
    for i in range(len(X)):
        value.append(X[i][dim])
    return vector(value)


cpdef axis_of_max_variance(list Y, int dim):
    cdef vector axis
    cdef vector max_axis
    cdef float max_var = 0
    cdef float axis_var
    cdef int idx = -1
    cdef list axes = []
    for j in range(dim):
        axis = vector_from_axis(Y, j)
        axis_var = axis.var()
        # axes.append(axis)
        if axis_var > max_var:
            max_var = axis_var
            max_axis = axis
    return idx, max_axis


cpdef list approximate_list(list v, tolerance=.1):
    cdef float value
    cdef int tol
    for i, value in enumerate(v):
        tol = len(str(round(1 / (tolerance * value))))
        v[i] = round(value, tol)
    return v


cpdef float approximate(float value, tolerance=.1):
    tol = len(str(round(1 / (value * tolerance))))
    return round(value, tol)


cpdef list around(list v):
    return [int(round(i)) for i in v]


cdef int cmp_func(const void* a, const void* b) nogil:
    cdef double a_v = (<double*>a)[0]
    cdef double b_v = (<double*>b)[0]
    if a_v < b_v:
        return -1
    elif a_v == b_v:
        return 0
    else:
        return 1


def sort_cpp(float[::1] a):
    # a must be c continuous (enforced with [::1])
    sort(&a[0], (&a[0]) + a.shape[0])
