from scipy.optimize import minimize
import time
import torch

from . import model

def obj_fcn__wrap(x_free, args): 
    x_free_torch = args['x_free_torch']
    
    x_free_torch.data.copy_(torch.from_numpy(x_free).data)
    loss_torch = model.obj_fcn(x_free_torch, args)
    loss = loss_torch.item()

    loss_torch.backward()
    grad_free = x_free_torch.grad.detach().cpu().clone().numpy()
    x_free_torch.grad.data.zero_()
    
    return loss, grad_free

def optimize__scipy(x_free, args,
                    opt_dict):    
    time_start = time.time()
    min_result = minimize(obj_fcn__wrap,
                          x_free,
                          args=args,
                          method=opt_dict['opt_method'],
                          jac=True,
                          hess=None,
                          hessp=None,
                          bounds=args['bounds_free'],
                          constraints=(),
                          tol=None,
                          callback=None,
                          options=opt_dict['opt_options'])
    time_end = time.time() 
    print('iterations:\t{:06d}'.format(min_result.nit))
    print('residual:\t{:0.8e}'.format(min_result.fun))
    print('success:\t{}'.format(min_result.success))
    print('message:\t{}'.format(min_result.message))
    print('time needed:\t{:0.3f} seconds'.format(time_end - time_start))
    return min_result
