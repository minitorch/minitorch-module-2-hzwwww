from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    vals = list(vals)
    x = vals[arg]
    
    vals[arg] = x - epsilon
    y_left = f(*vals)
    vals[arg] = x + epsilon
    y_right = f(*vals)
    
    return (y_right - y_left) / 2 / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.  
    from typing import Any, Callable, Deque, Dict, Iterable, List, Set, Tuple
    from collections import defaultdict, deque

    in_degree: Dict[int, int] = defaultdict(int)
    in_degree[variable.unique_id] = 0

    # Stack to keep track of nodes to visit, using doubly ended queue for O(1) pop and append
    stack: Deque[Variable] = deque([variable])
    visited: Set[int] = set([variable.unique_id])  # Keep track of visited nodes
    result: List[Variable] = []  # List to store the topological order

    # First pass: Calculate in-degrees and identify all nodes, using iterative DFS
    while stack:
        cur_var = stack.pop()

        # Explore the parents of the current variable, counting the incoming edges
        for var in cur_var.parents:
            # Skip constant variables since they do not have derivatives
            # Otherwise, increment the in-degree of the parent
            if not var.is_constant():
                in_degree[var.unique_id] += 1

                # If the parent has not been visited, add it to the stack
                if var.unique_id not in visited:
                    stack.append(var)
                    visited.add(var.unique_id)

    # Reset the stack and add the variable to the stack
    stack.append(variable)

    # Second pass: Topological sorting using zero in-degree nodes
    # Only add variable to the result when all its dependencies (i.e. parents) have been processed (in_degree = 0)
    while stack:
        cur_var = stack.pop()
        result.append(cur_var)

        for var in cur_var.parents:
            # If the variable is not a constant, decrement the number of incoming edges because the parent will be visited
            if not var.is_constant():
                in_degree[var.unique_id] -= 1

                # If the parent has zero incoming edges, add it to the stack to be visited
                if in_degree[var.unique_id] == 0:
                    stack.append(var)

    return result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    
    # Dictionary to store derivatives for each variable
    derivatives_dict = {variable.unique_id: deriv}

    top_sort = topological_sort(variable)

    # Iterate through the topological order and calculate the derivatives
    for curr_var in top_sort:
        if curr_var.is_leaf():
            continue

        # Get the derivatives of the current variable
        var_n_der = curr_var.chain_rule(derivatives_dict[curr_var.unique_id])

        # Accumulate the derivative for each parent of the current variable
        for var, deriv in var_n_der:
            if var.is_leaf():
                var.accumulate_derivative(deriv)
            else:
                if var.unique_id not in derivatives_dict:
                    derivatives_dict[var.unique_id] = deriv
                else:
                    derivatives_dict[var.unique_id] += deriv
    # print('---' * 10, deriv_dict)

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
