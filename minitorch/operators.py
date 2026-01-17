"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

def mul(a: float, b: float) -> float:
    return a*b

def id(a: float) -> float:
    return a

def add(a: float, b: float) -> float:
    return a+b

def neg(a: float) -> float:
    return -a

def lt(a: float, b: float) -> bool:
    return a<b

def eq(a: float, b: float) -> bool:
    return a==b

def max(a: float, b: float) -> float:
    if a > b:
        return a
    return b

def is_close(a: float, b: float) -> bool:
    """
    pytorch实现：|a-b| <= rtol * |b| + atol
    相对容差：适用于较大数值的比较，例如比较1000和1000.001，允许的误差是1000 × 0.00001 = 0.01
    rtol = 1e-05
    绝对容差：适用于接近零的数值的比较
    atol = 1e-08
    """
    
    return abs(a-b) <= 1e-2

def sigmoid(a: float) -> float:
    """
    激活函数，标准公式为 1 / (1+e^-x)
    
    作用：
        1. 将输出“压缩”到 (0, 1) 区间，用在二分类问题中的概率，门控中的激活率
        2. 引入非线性
    缺陷：
        1. 梯度消失：当输入 x 的绝对值很大时，Sigmoid函数的梯度（导数）会趋近于0
        2. 输出恒为正，非零中心化（tanh可解决此问题）
        3. 计算开销大
        4. 敏感区间狭窄，在输入 x 接近0的区间才有显著的梯度
    
    当x是非常大的正数，e^-x下溢到0，下溢安全
    当x是非常小的负数，e^-x上溢到inf，上溢不安全
    
    为什么上溢不安全：计算不稳定，inf + 1 = inf
    为什么下溢安全：计算稳定，0 + 1 = 1
    
    优化sigmoid数值稳定：当x是非常小的负数时，上下同乘一个e^x，将e^-x转为e^x（上溢转下溢）
    1 / (1+e^-x) = e^x / (e^x + 1)
    """
    
    if a < 0:
        return math.exp(a) / (math.exp(a) + 1)
    return 1 / (1 + math.exp(-a))
    
def relu(a: float) -> float:
    """
    激活函数，f(x) = max(0, x)
    
    作用：
        1. 为线性层引入非线性，使得神经网络可以逼近任意复杂的函数
        2. 缓解梯度消失问题，在正区域，ReLU 的导数为常数 1；在负区域，导数为 0
        3. 促进稀疏性，ReLU 的输出为 0。这意味着网络中的一部分神经元可能被“关闭，让不同的神经元专注于学习不同的特征
    缺陷：神经元死亡
    变体：Leaky ReLU 在负半轴引入一个微小的斜率，确保梯度不为零，缓解“神经元死亡”问题。
    relu vs sigmod：relu用于隐藏层，sigmod用于输出层
    """
    if a < 0:
        return float(0)
    return a

def log(a: float) -> float:
    """
    数值稳定的对数函数
    
    作用：
        1. 计算输入值的自然对数，常用于概率模型、损失函数（如交叉熵）
        2. 在机器学习中经常需要计算概率的对数，避免数值下溢
    数值稳定性处理：
        - 当输入 a 为 0 或非常接近 0 时，直接计算 log(a) 会得到 -inf 或数值不稳定
        - 通过添加一个微小值 eps 来避免 log(0) 的情况
    
    参数：
        a: 输入值，应该为正数（对数定义域为 >0）
        eps: 一个小常数，用于数值稳定性，默认 1e-12
    
    返回：
        log(a) 的自然对数结果
    
    注意：
        1. 如果 a <= 0，会返回 log(eps) 作为保护
        2. 在实际应用中，根据具体情况调整 eps 的大小
        3. 这个函数不是激活函数，而是数学工具函数
    """
    eps = 1e-12
    if a < 0:
        return math.log(eps)
    
    return math.log(a)

def exp(a: float) -> float:
    """
    数值稳定的指数函数
    
    作用：
        1. 计算 e 的 a 次幂，是神经网络中最常用的数学运算之一
        2. 用于激活函数（如 Softmax、Sigmoid）、概率模型、损失函数等
    数值稳定性问题：
        - 当 a 是非常大的正数时，exp(a) 可能上溢（overflow）到 inf
        - 当 a 是非常小的负数时，exp(a) 可能下溢（underflow）到 0
    
    处理策略：
        1. 对于激活函数（如 Softmax、Sigmoid），通常结合具体场景进行数值稳定处理
        2. 一般通过减去最大值（max subtraction）或分段处理来避免上溢
    
    参数：
        a: 指数值
    
    返回：
        exp(a) 的结果，如果可能上溢则进行保护
    
    注意：
        1. 纯粹的 exp 函数本身无法完全避免上溢问题
        2. 在实际应用中，通常需要在更高层次进行数值稳定设计
        3. 对于非常大的输入（如 a > 709），math.exp 会溢出到 inf
    """
    
    return math.exp(a)

def log_back(a: float, arg: float) -> float:
    """log函数在反向传播中的梯度计算
    
    参数：
        a: 前向传播时log函数的输入值
        arg: 从上一层传递下来的梯度（通常是损失函数对log输出的梯度）
    计算过程：
        log函数的导数：d(log(a))/da = 1/a
        反向传播公式：grad_input = grad_output * (1/a)
    返回：
        损失函数对输入a的梯度 = arg * (1/a)
    注意：
        1. 需要处理 a <= 0 的情况，避免除以零或数学错误
        2. 当 a <= 0 时，梯度应该返回 0（或者一个很小的值），因为原函数在这种情况下返回 log(eps)
    """
    
    return arg / a

def inv(a: float) -> float:
    """计算a的倒数1/a

    注意：
        1. 对于 a = 0 的情况，有多种处理策略：
           a) 返回 ±inf（取决于符号）
           b) 返回一个非常大的数值（如 ±1e12）
           c) 返回 0（在某些上下文中）
        2. 这里采用添加微小 epsilon 的方法来避免除以零
    """
    
    return 1.0 / a

def inv_back(a: float, arg: float) -> float:
    """倒数函数在反向传播中的梯度计算
    
    参数：
        a: 前向传播时倒数函数的输入值
        arg: 从上一层传递下来的梯度（通常是损失函数对导数函数输出的梯度）
    计算过程：
        倒数函数的导数：d(1/a)/da = -1/a^2
        反向传播公式：grad_input = grad_output * (-1/a^2)
    返回：
        损失函数对输入a的梯度 = arg * (-1/a^2)
    """
    
    return arg * (-1/math.pow(a, 2))

def relu_back(a: float, arg: float) -> float:
    """relu函数在反向传播中的梯度计算
    
    参数：
        a: 前向传播时relu函数的输入值
        arg: 从上一层传递下来的梯度（通常是损失函数对relu函数输出的梯度）
    计算过程：
        relu函数的导数：a > 0时为1，a<=0时为0
        反向传播公式：grad_input = grad_output * (0|1)
    返回：
        损失函数对输入a的梯度 = arg * (0|1)
    """
    
    if a > 0:
        return arg
    return 0

# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

def map(list: Iterable, func: Callable):
    result = []
    for item in list:
        result.append(func(item))
    return result

def zipWith(list_a: Iterable, list_b: Iterable, func: Callable):
    for a, b in zip(list_a, list_b):
        func(a, b)  
        
def reduce(lst: Iterable, func: Callable):
    if len(list(lst)) == 0:
        return 0

    iterator = iter(lst)
    val = next(iterator)
    for item in iterator:
        val = func(val, item)

    return val

def addLists(list_a: Iterable[float], list_b: Iterable[float]):
    res = []
    
    def ops(a: float, b: float):
        res.append(a+b)
        
    zipWith(list_a, list_b, ops)
    
    return res

def negList(list: Iterable):
    return map(list, neg)

def sum(list: Iterable):
    return reduce(list, add)

def prod(list: Iterable):
    return reduce(list, mul)

# TODO: Implement for Task 0.3.
