import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1) # set the seed for generating random numbers


'''
# Introduction ======================================================================================
# ------------------------------------------------------------------------------
# Create a torch.Tensor object with the given data.  It is a 1D vector
V_data = [1., 2., 3.]
V = torch.Tensor(V_data)
print(V)
# Index into V and get a scalar
print(V[0])

# Creates a matrix
M_data = [[1., 2., 3.], [4., 5., 6]]
M = torch.Tensor(M_data)
print(M)
# Index into M and get a vector
print(M[0])

# Create a 3D tensor of size 2x2x2.
T_data = [[[1., 2.], [3., 4.]],
          [[5., 6.], [7., 8.]]]
T = torch.Tensor(T_data)
print(T)
# Index into T and get a matrix
print(T[0])

# ------------------------------------------------------------------------------
# Create a tensor with random data and the supplied dimensionality with torch.randn()
x = torch.randn((3, 4, 5))
print(x)

# Operate on Tensors
x = torch.Tensor([1., 2., 3.])
y = torch.Tensor([4., 5., 6.])
z = x + y
print(x,y,z)

# One helpful operation that we will make use of later is concatenation
# By default, it concatenates along the first axis (concatenates rows)
x_1 = torch.randn(2, 5)
y_1 = torch.randn(3, 5)
# second arg specifies which axis to concat along
z_1 = torch.cat([x_1, y_1], 0)
print(z_1)

# Concatenate columns:
x_2 = torch.randn(2, 3)
y_2 = torch.randn(2, 5)
z_2 = torch.cat([x_2, y_2], 1)
print(z_2)

# Use view() method to reshape a tensor
# Many NN components expect the inputs to have a certain shape
x = torch.randn(2, 3, 4)
print(x)
print(x.view(2, 12))  # Reshape to 2 rows, 12 columns
# Same as above. If one of the dimensions is -1, its size can be inferred
print(x.view(2, -1))
'''


# Computating graphs and automatic differentiation ================================================
# A computation graph is simply a specification of how your data is combined to give you output.
# The graph totally specifies what parameters were involved with which operations..
# The graph contains enough information to compute derivatives.

# The Variable class keeps track of how it was created. Lets see it in action.
# If you want the error from your loss function to backpropagate to a component 
# of your network, you MUST NOT break the Variable chain from that component to 
# your loss Variable. If you do, the loss will have no idea your component exists 
# and its parameters can’t be updated.
x = torch.randn((2, 2))
y = torch.randn((2, 2))
z = x + y  # These are Tensor types, and backprop would not be possible

var_x = autograd.Variable(x, requires_grad=True)
var_y = autograd.Variable(y, requires_grad=True)
# var_z contains enough information to compute gradients
var_z = var_x + var_y
print(var_z.grad_fn)

# Get the wrapped Tensor object out of var_z..., this breaks the "chain" !
var_z_data = var_z.data  
# Re-wrap the tensor in a new variable
new_var_z = autograd.Variable(var_z_data)

# ... does new_var_z have information to backprop to x and y?
# NO!
print(new_var_z.grad_fn)
# And how could it?  We yanked the tensor out of var_z (that is
# what var_z.data is).  This tensor doesn't know anything about
# how it was computed.  We pass it into new_var_z, and this is all the
# information new_var_z gets.  If var_z_data doesn't know how it was
# computed, theres no way new_var_z will.
# In essence, we have broken the variable away from its past history













