import os
import numpy as np 
import matplotlib.pyplot as plt 

# set random seed to make experiment reproducible 
np.random.seed(1)

#################### Task 1 ###################
"""
Implement the random search function
Below we have a Python wrapper providing a skeleton for 
your production of of the random local search algorithm. 
All parts marked "TO DO" are for you to construct.
Notice that the history of function evaluations returned is called cost_history. 
This is because - in the context of machine learning / deep learning - 
mathematical functions are often referred to as cost or loss functions.
"""
def random_search(g, w, alpha=1, max_its=50, num_samples=1000, diminishing_steplength=False):
    """
    params:
    - g: the function to optimize
    - w: a numpy array, the parameters of g
    - a: learning rate (alpha)
    - K: the maximum number of iterations
    - P: the number of directions sampled in each iteration
    - diminish_steplength: if True, use diminishing learning rate

    returns: 
    - weight_history: a list of the weight values 
    - cost_history: a list of cost values evaluated with g 
    """
    weight_history = [w]  # container for weight history
    
    cost_history = [g(w)] # container for corresponding cost function history
    
    for k in range(1, max_its+1):
        
        if diminishing_steplength:

            alpha = 1 / k

        r = np.random.randn(num_samples, w.size)
        
        d = r / np.linalg.norm(r, axis=1)[:, np.newaxis]
        
        weights = weight_history[k-1] + alpha * d
        
        costs = np.array([g(weight) for weight in weights])
        
        argmin = costs.argmin()
        
        progress = costs[argmin] < cost_history[k-1]
        
        weight_history.append(weights[argmin] if progress else weight_history[k-1])
        
        cost_history.append(costs[argmin] if progress else cost_history[k-1])
    
    print(f"Random search finished with K={max_its} iterations")
    
    return weight_history, cost_history

def run_task1():
    
    print("Run task 1 ....")
    
    g = lambda x: x ** 2
    
    w = np.array([-2])
    
    weight_history, cost_history = random_search(g, w, num_samples=5, max_its=5)
    
    assert np.isclose(cost_history[-1], 0), f"The min cost should be 0, not {cost_history[-1]}"
    
    assert np.isclose(weight_history[-1], 0), f"The min weight should be 0, not {weight_history[-1]}"
    
    print(weight_history)

    print(cost_history)

    print("Task 1 finished.")

#################### Task 2 ###################
"""
In task 2, you should apply the implemented random search function to optimize 
the function described in the homework assignment. 
All parts marked "TODO" are for you to construct.
"""
# plot function 
def plot_cost_history(cost_history): 
    
    plt.figure() 
    
    plt.plot(np.arange(1, len(cost_history)+1), cost_history, marker='o')
    
    plt.xlabel("k")
    
    plt.ylabel('cost')
    
    plt.savefig("cost_history.png") 
    
    # plt.show()

def con_func(w): 
    """
    The function to be minimized.
    Params: 
    - w: the parameters of the function 
    Returns: 
    - cost: the value of the function
    """
    return 100 * (w[1] - w[0] ** 2) ** 2 + (w[0] - 1) ** 2 

def run_task2(): 

    print("Run task 2 ....")
    
    plot_cost_history(random_search(con_func, np.array([-2,-2]))[1])
    
    print("Task 2 finished.")

#################### Task 3 ###################
"""
In task 3, you should improve the random search function to 
allow dinimishing learning rate. 
Then apply the random search function to the function again 
All parts marked "TODO" are for you to construct.

After you get the cost_history, you should plot the cost_history with a fixed learning rate
 and the dinimishing learning rate in the same figure to compare them.
"""
def compare_cost_history(costs_fixed, costs_diminished):
    
    plt.figure()
    
    plt.plot(np.arange(1, len(costs_fixed) + 1), costs_fixed, 'k-', marker="o", label="with fixed steplength")
    
    plt.plot(np.arange(1, len(costs_diminished) + 1), costs_diminished, "r-", marker='o', label="with diminishing steplength")
    
    plt.xlabel("k")
    
    plt.ylabel("cost")
    
    plt.legend()
    
    plt.savefig("comparison.png")


def compare_contour(weights_fixed, weights_dinimished): 
    
    delta = 0.001
    
    x = np.arange(-5.0, 5.0, delta)
    
    y = np.arange(-5.0, 5.0, delta)
    
    X, Y = np.meshgrid(x, y)
    
    Z = 100 * (Y - X * X)**2 + (X - 1)**2 
    
    fig, ax = plt.subplots()
    
    CS = ax.contour(X, Y, Z)
    
    ax.clabel(CS, inline=True, fontsize=10)
    
    ax.set_title('Contour') 

    # plot weights on top of it 
    
    w1 = [x[0] for x in weights_fixed]
    
    w2 = [x[1] for x in weights_fixed]
    
    ax.plot(w1, w2, 'k-', marker='o')

    w1 = [x[0] for x in weights_dinimished]
    
    w2 = [x[1] for x in weights_dinimished]
    
    ax.plot(w1, w2, 'r-', marker='o')

    fig.savefig("contour.png")

def run_task3():

    print("Run task 3 ...") 

    # TODO: fill in the code
    # You coudl use "compare_cost_history" and "compare_contour" to 
    # produce figures used for the report. 

    w_fixed, c_fixed = random_search(con_func, np.array([-2,-2]))

    w_diminished, c_diminished = random_search(con_func, np.array([-2,-2]), diminishing_steplength=True)

    # Uncomment the following lines one at a time to reproduce report figures.
    # compare_cost_history(c_fixed[10:], c_diminished[10:])
    # compare_cost_history(c_fixed, c_diminished)
    # compare_contour(w_fixed, w_diminished)

    print(f"Fixed Steplength Final Cost: {c_fixed[-1]}")

    print(f"Diminishing Steplength Final Cost: {c_diminished[-1]}")

    print("Task 3 finished.") 

if __name__ == '__main__':

    run_task1() 

    run_task2()

    run_task3()