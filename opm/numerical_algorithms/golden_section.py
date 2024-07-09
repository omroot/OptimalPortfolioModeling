
from typing import Callable, Tuple, Any
from math import log, ceil

def golden_section(obj: Callable[..., float],
                   a: float,
                   b: float,
                   **kargs: Any) -> Tuple[float, float]:
    """
    Golden section method for optimization.

    Args:
    obj (Callable[..., float]): Objective function to optimize.
    a (float): Left endpoint of the initial interval.
    b (float): Right endpoint of the initial interval.
    **kargs: Additional keyword arguments:
        - minimum (bool, optional): If True, find minimum; if False, find maximum.
        - args (tuple, optional): Additional arguments for obj function.

    Returns:
    Tuple[float, float]: x value that optimizes the function obj within [a, b], and corresponding optimized value.

    Notes:
    - Maximum is found if kargs['minimum']==False is passed.
    """



    tol, sign, args = 1.0e-9, 1, None
    if 'minimum' in kargs and kargs['minimum'] == False:
        sign = -1
    if 'args' in kargs:
        args = kargs['args']

    # r = (np.sqrt(5) - 1)/2
    r = 0.618033989
    c = 1.0 - r

    num_iter = int(ceil(  log(tol / abs(b - a)) / log (1/r)   )  )

    # Initialize
    x1 = r * a + c * b
    x2 = c * a + r * b
    f1 = sign * obj(x1, *args)
    f2 = sign * obj(x2, *args)

    # Loop
    for i in range(num_iter):
        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = c * a + r * b
            f2 = sign * obj(x2, *args)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = r * a + c * b
            f1 = sign * obj(x1, *args)

    if f1 < f2:
        return x1, sign * f1
    else:
        return x2, sign * f2
