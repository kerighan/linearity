#!python
#cython: language_level=3
#cython.wraparound=False
#cython.boundscheck=False
#cython.nonecheck=False
import math
cimport cython


cdef class vector(object):
    cdef public list value
    cdef public Py_ssize_t size

    def __init__(self, list v):
        cdef Py_ssize_t i
        self.size = len(v)
        self.value = v
    
    def normalize(self):
        cdef float vec_norm = norm(self)
        if vec_norm == 0:
            raise ValueError("cannot normalize zero vector")
        self.value = [v / vec_norm for v in self.value]
        return self
    
    def norm(self):
        return norm(self)

    cpdef sum(self):
        cdef list value = self.value
        cdef Py_ssize_t v_len = len(value)
        cdef float s = 0
        cdef Py_ssize_t i
        for i in range(v_len):
            s += value[i]
        return s

    def distance_from(self, item):
        if isinstance(item, vector):
            return norm(self - item)
        elif isinstance(item, hyperplane):
            return math.fabs(dot(self - item.position, item.normal))

    def __list__(self):
        return self.value

    def is_null(self):
        return norm(self) == 0

    def __eq__(self, v):
        return all(
            (self.value[i] == v.value[i]
            for i in range(self.size)))

    cpdef vector add(self, vector v):
        cdef Py_ssize_t size = v.size
        cdef Py_ssize_t i
        cdef list vec = []
        cdef list self_value = self.value
        cdef list v_value = v.value
        for i in range(size):
            vec.append(v_value[i] + self_value[i])
        return vector(vec)

    cpdef vector sub(self, vector v):
        cdef Py_ssize_t size = v.size
        cdef Py_ssize_t i
        cdef list vec = []
        cdef list self_value = self.value
        cdef list v_value = v.value
        for i in range(size):
            vec.append(v_value[i] - self_value[i])
        return vector(vec)

    def __add__(self, vector v):
        return self.add(v)
    
    def __sub__(self, vector v):
        return self.sub(v)
    
    def __setitem__(self, key, value):
        self.value[key] = value

    def __getitem__(self, key):
        return self.value[key]

    def __rmul__(self, v):
        return self.__mul__(v)
    
    cpdef vector divide_by_scalar(self, float scalar):
        cdef Py_ssize_t size = self.size
        cdef Py_ssize_t i
        cdef list vec = []
        cdef list self_value = self.value
        for i in range(size):
            vec.append(self_value[i] / scalar)
        return vector(vec)

    def __truediv__(self, v):
        if isinstance(v, vector):
            return vector([
                self.value[i] / v.value[i]
                for i in range(self.size)
            ])
        else:
            return self.divide_by_scalar(v)

    def __repr__(self):
        return str(self.value)

    def __ge__(self, hyperplane h):
        return h.side_of(self)


cpdef ones(int dim):
    cdef list v = []
    cdef Py_ssize_t i
    for i in range(dim):
        v.append(1)
    return vector(v)


cpdef zeros(dim):
    cdef list v = []
    cdef Py_ssize_t i
    for i in range(dim):
        v.append(0)
    return vector(v)


cpdef vector random(int dim):
    import random as rd
    cdef list v = []
    cdef Py_ssize_t i
    for i in range(dim):
        v.append(rd.random())
    return vector(v)


cdef class hyperplane(object):
    cdef public vector normal
    cdef public vector position

    def __init__(self, vector normal, vector position=None, int to_norm=True):
        if to_norm:
            self.normal = normalize(normal)
        else:
            self.normal = normal
        if position is None:
            self.position = zeros(normal.size)
        else:
            self.position = position

    cpdef int side_of(self, vector v):
        cdef list v_value = v.value
        cdef list p_value = self.position.value
        cdef list n_value = self.normal.value
        cdef float dot_value = 0
        cdef Py_ssize_t size = v.size
        cdef Py_ssize_t i
        for i in range(size):
            dot_value += (v_value[i] - p_value[i]) * n_value[i]
        if dot_value >= 0:
            return 1
        return 0

    cpdef list to_list(self):
        return [self.normal.value, self.position.value]


cpdef hyperplane hyperplane_from_pair(vector a, vector b):
    cpdef vector position = center(a, b)
    cpdef vector normal = b - a
    return hyperplane(normal, position)


cpdef float mean(a):
    cdef Py_ssize_t i
    cdef Py_ssize_t size = a.size
    cdef list value = a.value
    cdef float res = 0
    for i in range(size):
        res += value[i]
    return res / size


cpdef float dot(vector a, vector b):
    cdef list a_value = a.value
    cdef list b_value = b.value
    cdef float dot_value = 0
    cdef Py_ssize_t size = len(a_value)
    cdef Py_ssize_t i
    for i in range(size):
        dot_value += a_value[i] * b_value[i]
    return dot_value


cpdef float norm(vector a):
    return math.sqrt(dot(a, a))


cpdef vector normalize(vector a):
    cdef float vec_norm = norm(a)
    return vector([v / vec_norm for v in a.value])


cpdef vector center(vector a, vector b):
    cdef list val_a = a.value
    cdef list val_b = b.value
    cdef list vec = []
    cdef Py_ssize_t size = a.size
    cdef Py_ssize_t i
    for i in range(size):
        vec.append((val_a[i] + val_b[i]) / 2.)
    return vector(vec)


cpdef list approximate_list(list vector, tolerance=.1):
    cdef float value
    cdef int tol
    for i, value in enumerate(vector):
        tol = len(str(round(1 / (tolerance * value))))
        vector[i] = round(value, tol)
    return vector


cpdef separate_vectors_from_hyperplane(list X, hyperplane hyp):
    cdef list upper = []
    cdef list lower = []
    cdef tuple vector
    for vector in X:
        if hyp.side_of(vector[0]):
            upper.append(vector)
        else:
            lower.append(vector)
    return upper, lower
