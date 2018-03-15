




# Loss functions ##########################################################################################
# --------------------------------------------------------------------------                            
# Creates a criterion that measures the mean absolute value of the element-wise difference between input x and target y                                
criterion = torch.nn.L1Loss(size_average=True, reduce=True)                                              
# --------------------------------------------------------------------------                             
# Creates a criterion that measures mean squared error between n elements in the input x and target y    
criterion = torch.nn.MSELoss(size_average=True, reduce=True)
# --------------------------------------------------------------------------                             
# This criterion combines LogSoftMax and NLLLoss in one single class                                     
criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=True, ignore_index=-100, reduce=True)                                             
# --------------------------------------------------------------------------                             
# The negative log likelihood loss. It is useful to train a classification problem with C classes        
criterion = torch.nn.NLLLoss(weight=None, size_average=True, ignore_index=-100, reduce=True)             
# --------------------------------------------------------------------------                             
# Negative log likelihood loss with Poisson distribution of target
criterion = torch.nn.PoissonNLLLoss(log_input=True, full=False, size_average=True, eps=1e-08, reduce=True)
# --------------------------------------------------------------------------                             
# The Kullback-Leibler divergence Loss
criterion = torch.nn.KLDivLoss(size_average=True, reduce=True)
# --------------------------------------------------------------------------                             
# Creates a criterion that measures the Binary Cross Entropy between the target and the output
criterion = torch.nn.BCELoss(weight=None, size_average=True, reduce=True)
# --------------------------------------------------------------------------                             
# This loss combines a Sigmoid layer and the BCELoss in one single class
criterion = torch.nn.BCEWithLogitsLoss(weight=None, size_average=True, reduce=True)
# --------------------------------------------------------------------------                             
# Creates a criterion that measures the loss given inputs x1, x2, two 1D mini-batch Tensor`s, and a label 1D mini-batch tensor `y with values (1 or -1)
criterion = torch.nn.MarginRankingLoss(margin=0, size_average=True)[source]
# --------------------------------------------------------------------------                             
# Measures the loss given an input tensor x and a labels tensor y containing values (1 or -1)
criterion = torch.nn.HingeEmbeddingLoss(margin=1.0, size_average=True, reduce=True)
# --------------------------------------------------------------------------                             
# Creates a criterion that optimizes a multi-class multi-classification hinge loss
criterion = torch.nn.MultiLabelMarginLoss(size_average=True, reduce=True)
# --------------------------------------------------------------------------                             
# Creates a criterion that uses a squared term if the absolute element-wise error falls below 1 and an L1 term otherwise
criterion = torch.nn.SmoothL1Loss(size_average=True, reduce=True)
# --------------------------------------------------------------------------                             
# Creates a criterion that optimizes a two-class classification logistic loss between input tensor x and target tensor y (containing 1 or -1)
criterion = torch.nn.SoftMarginLoss(size_average=True, reduce=True)
# --------------------------------------------------------------------------                             
# Creates a criterion that optimizes a multi-label one-versus-all loss based on max-entropy
criterion = torch.nn.MultiLabelSoftMarginLoss(weight=None, size_average=True, reduce=True)
# --------------------------------------------------------------------------                             
# Creates a criterion that measures the loss given an input tensors x1, x2 and a Tensor label y with values 1 or -1
criterion = torch.nn.CosineEmbeddingLoss(margin=0, size_average=True)
# --------------------------------------------------------------------------                             
# Creates a criterion that optimizes a multi-class classification hinge loss
criterion = torch.nn.MultiMarginLoss(p=1, margin=1, weight=None, size_average=True)
# --------------------------------------------------------------------------                             
# Creates a criterion that measures the triplet loss given an input tensors x1, x2, x3 and a margin with a value greater than 0
criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-06, swap=False)
# Optimizations ###########################################################################################
# --------------------------------------------------------------------------
# Base class for all optimizers
optimizer = torch.optim.Optimizer(params, defaults)
# --------------------------------------------------------------------------
# Implements Adadelta algorithm
optimizer = torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
# --------------------------------------------------------------------------
# Implements Adagrad algorithm
optimizer = torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0)
# --------------------------------------------------------------------------
# Implements Adam algorithm
optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# --------------------------------------------------------------------------
# Implements lazy version of Adam algorithm suitable for sparse tensors
optimizer = torch.optim.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
# --------------------------------------------------------------------------
# Implements Adamax algorithm (a variant of Adam based on infinity norm)
optimizer = torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# --------------------------------------------------------------------------
# Implements Averaged Stochastic Gradient Descent
optimizer = torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
# --------------------------------------------------------------------------
# Implements L-BFGS algorithm
optimizer = torch.optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
# --------------------------------------------------------------------------
# Implements RMSprop algorithm
optimizer = torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
# --------------------------------------------------------------------------
# Implements the resilient backpropagation algorithm
optimizer = torch.optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
# --------------------------------------------------------------------------
# Implements stochastic gradient descent (optionally with momentum)
optimizer = torch.optim.SGD(nnModel.parameters(), lr=learning_rate, momentum=0, dampening=0, weight_decay=0, nesterov=False)
# Dynamic learning rate ###################################################################################
# --------------------------------------------------------------------------
# Sets the learning rate of each parameter group to the initial lr times a given function
lambda1 = lambda epochs: epochs // 30
lambda2 = lambda epochs: 0.95 ** epochs
scheduler = torch.optim.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
# --------------------------------------------------------------------------
# Sets the learning rate of each parameter group to the initial lr decayed by gamma every step_size epochs
scheduler = torch.optim.StepLR(optimizer, step_size=30, gamma=0.1)
# --------------------------------------------------------------------------
# Set the learning rate of each parameter group to the initial lr decayed by gamma once the number of epoch reaches one of the milestones
scheduler = torch.optim.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
# --------------------------------------------------------------------------
# Set the learning rate of each parameter group to the initial lr decayed by gamma every epoch
scheduler = torch.optim.ExponentialLR(optimizer, gamma, last_epoch=-1)
# --------------------------------------------------------------------------
# Set the learning rate of each parameter group using a cosine annealing schedule
scheduler = torch.optim.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
# --------------------------------------------------------------------------
# Reduce learning rate when a metric has stopped improving
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)










