import torch

def MyGradCheck_Wiener(function, inputs, eps=1e-8, tol=1e-6):
    y, k, lam, weights = inputs;

    output = function.apply(y, k, lam, weights);
    grad_output = output.new(output.shape).uniform_();

    y_numerical_grad = ((function.apply(y.detach()+eps, k.detach(), lam.detach(), weights.detach()) - 
    function.apply(y.detach()-eps, k.detach(), lam.detach(), weights.detach()))/(2*eps))*grad_output;

    lam_numerical_grad = torch.sum(((function.apply(y.detach(), k.detach(), lam.detach()+eps, weights.detach()) - 
    function.apply(y.detach(), k.detach(), lam.detach()-eps, weights.detach()))/(2*eps))*grad_output);

    weights_numerical_grad = weights.new(weights.shape).fill_(1);
    weights_ench_plus = weights.detach().clone();
    weights_ench_minus = weights.detach().clone();
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            for l in range(weights.shape[2]):
                for m in range(weights.shape[3]):
                    weights_ench_plus[i,j,l,m] = weights_ench_plus[i,j,l,m] + eps;
                    weights_ench_minus[i,j,l,m] = weights_ench_minus[i,j,l,m] - eps;
                    weights_numerical_grad[i,j,l,m] = torch.sum(((function.apply(y.detach(), k.detach(), lam.detach(), weights_ench_plus) - 
                    function.apply(y.detach(), k.detach(), lam.detach(), weights_ench_minus))/(2*eps))*grad_output);
                    weights_ench_plus[i,j,l,m] = weights_ench_plus[i,j,l,m] - eps;
                    weights_ench_minus[i,j,l,m] = weights_ench_minus[i,j,l,m] + eps;
    if y.grad is not None:
        y.grad.detach_()
        y.grad.data.zero_()
    if lam.grad is not None:
        lam.grad.detach_()
        lam.grad.data.zero_()
    if weights.grad is not None:
        dweights.grad.detach_()
        dweights.grad.data.zero_()

    output = function.apply(y, k, lam, weights);
    (y_analytical_grad, lam_analytical_grad, weights_analytical_grad) = torch.autograd.grad(output,  (y, lam, weights), grad_output, create_graph=True);

    check_y = torch.mean(y_analytical_grad - y_numerical_grad);
    check_lam = torch.abs(lam_analytical_grad - lam_numerical_grad);
    check_weights = torch.mean(weights_analytical_grad - weights_numerical_grad);
    print("Mean input grad difference:")
    print(check_y)
    print("\n")

    print("Lambda grad difference:")
    print(check_lam)
    print("\n")

    print("Mean weights grad difference:")
    print(check_weights)
    
    if (torch.abs(check_y)) < tol and (torch.abs(check_lam)) < tol and (torch.abs(check_weights)) < tol:
        return True
    else:
        return False