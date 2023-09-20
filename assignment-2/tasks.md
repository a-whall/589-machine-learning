# Assignment 2: Gradient Descent

# Task 1: Implement the gradient descent algorithm

Implement the gradient descent algorithm. Run gradient descent to minimize the following function: `g(w) = 1/50 * (w ** 4 + w ** 2 + 10 * w)`. Start with an initial point `w_0 = 2` and run for 1000 iterations. Make three separate runs using each of the step-length values `a = { 1, 0.1, 0.01 }`. Compute the derivative of this function by hand, and implement it (as well as the function itself) in Python using NumPy. Report the value of the function and its derivative at `w_0 = 2`. Plot the resulting cost function history plot of each run in a single figure to compare their performance. Also report which step-length value works best for this particular function and initial point.

A skeleton of the desired algorithm is in hw2.py. All parts marked TODO are for you to construct.

# Task 2: Compare fixed and diminishing steplengths with gradient descent

In this exercise you will compare a fixed steplength scheme and a diminishing step-length rule to minimize the function `g(w) = |w|`. Notice that this function has a single global minimum at `w = 0` and a derivative defined (everywhere but at `w = 0`). `g'(w) = +1, if w > 0; -1, if w < 0` which makes he use of any fixed step-length schema problematic for gradient descent.

You will make two runs of 20 steps of gradient descent each initialized at the point `w_0 = 2`, the first with a fixed step-length rule of `a = 0.5` for each and every step, and the second using the diminishing step-length rule `a = 1/k`.

Compute the cost function history associated with each desired run and plot both to compare. In this task you should use ![Jax]() to automatically compute the gradients of the given function.

A skeleton of the desired algorithm is provided in hw2.py. All parts marked TODO are for you to construct.

# Deliverables

- The completed source code hw2.py

- A PDF report

The source code should be able to run by executing the command `python hw2.py`. The PDF report should include:

- Task 1:
  - Value of the cost function (`g(w)`) at `w_0 = 2`
  - Value of the derivative of the cost function at `w_0 = 2`
  - Single figure containing the cost function history of three runs using the different step-length values.
  - Report which step-length works best for this particular function and initial point.
- Task 2:
  - Single figure containing the cost function history of two runs using a fixed step-length and a diminishing step-length.
