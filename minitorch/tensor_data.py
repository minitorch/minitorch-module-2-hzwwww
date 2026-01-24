from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        Position in storage
    """

    # TODO: Implement for Task 2.1.
    idx = 0
    for i in range(len(index)):
        idx += index[i] * strides[i]
    
    return idx


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """
    Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    strides = strides_from_shape(shape)
    for i in range(len(shape)):
        out_index[i] = ordinal // strides[i]
        ordinal %= strides[i]


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """
    Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
        None
    """
    # TODO: Implement for Task 2.2.
    
    """boradcast之后的tensor形状更大，但是底层数据用的还是小tensor
    dim不存在，直接忽略，(3, 2) of (2) => (i, j) of (j)
    dim=1，直接返回0，(3, 2, 4) of (2, 1) => (i, j, k) => (j, 0)
    dim相等，直接使用原始index
    """
    for i in range(len(shape)):
        out_index[i] = 0 if shape[i] == 1 else big_index[len(big_shape)-len(shape)+i]


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """
    Broadcast two shapes to create a new union shape.

    Args:
        shape1 : first shape
        shape2 : second shape

    Returns:
        broadcasted shape

    Raises:
        IndexingError : if cannot broadcast
    """
    # TODO: Implement for Task 2.2.
    
    """返回一个shape，保证与shape1和shape2符合broadcast原则
    原则：从右往左匹配shape1和shape2的每个dim，dim必须相等，或者为1，或者不存在
    举例：
        维度为1, a，则让dim=1的沿该维度复制a遍，[1,2,3] + [1] => [1,2,3] + [1,1,1]
        维度为0, a，则让dim=0的将前一个维度的数据延该维度复制a遍，[[1,2,3], [4,5,6]] + [[7,8,9]] => [[1,2,3], [4,5,6]] + [[7,8,9], [7,8,9]]
    """
    newShape = []
    i, j = len(shape1)-1, len(shape2)-1
    while i>=0 and j >=0:
        dim1 = shape1[i]
        dim2 = shape2[j]
        
        if dim1 == dim2:
            newShape.append(dim1)
        elif dim1 == 1:
            newShape.append(dim2)
        elif dim2 == 1:
            newShape.append(dim1)
        else:
            raise IndexingError(f'cant union {shape1} and {shape2}')
        
        i-=1
        j-=1
        
    # 填充左侧多余的维度
    while i >= 0:
        newShape.append(shape1[i])
        i-=1
    while j >= 0:
        newShape.append(shape2[j])
        j-=1
    
    return tuple(reversed(newShape))
    


def strides_from_shape(shape: UserShape) -> UserStrides:
    # layout = [1]
    # offset = 1
    # for s in reversed(shape):
    #     layout.append(s * offset)
    #     offset = s * offset
    # return tuple(reversed(layout[:-1]))
    # （3,4,5,1）
    stride = 1
    result = []
    for dim in reversed(shape):
        result.append(stride)
        stride *= dim
    return tuple(reversed(result))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        if isinstance(index, tuple):
            aindex = array(index)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """
        Permute the dimensions of the tensor.

        Args:
            order (list): a permutation of the dimensions

        Returns:
            New `TensorData` with the same storage and a new dimension order.
        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        # TODO: Implement for Task 2.1.
        """
        a[i, j] = a.permute(1, 0)[j, i]        
        
        examples:
            shape(3, 4),stride(4, 1),y(i, j) = (4i+j)
            shape(4, 3),stride(a, b),y(j, i) = (aj+bi) = (4i+j) => a=1,b=4 
            shape(3, 4, 5),stride(20, 5, 1), y(i, j, k) = (20i+5j+k)
            shape(4, 5, 3),stride(a, b,  c), y(j, k, i) = (aj+bk+ci) = (20i+5j+k) => a=5,b=1,c=20
            
        summary:
            permute(1, 2, 0)
                shape(i, j, k) => shape(j, k, i)
                stride(a, b, c) => stride(b, c, a)
        """
        
        shape = [self.shape[i] for i in order]
        strides = [self.strides[i] for i in order]
        
        return TensorData(self._storage, tuple(shape), tuple(strides))

    def to_string(self) -> str:
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
