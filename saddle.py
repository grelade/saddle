import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import itertools

# draw random k orthogonal N-vectors using pytorch
def rand_orthogonal(N,k):
    m = torch.randn(N,N)
    mat = m + torch.t(m)
    #return np.linalg.eig(mat)[1][:k]
    return torch.t(torch.symeig(mat,eigenvectors=True).eigenvectors[:k])

# Rastrigin potential function, nontrivial behaviour for q in (-5,5)
def rastrigin(q):
    return torch.sum(10+q*q-10*torch.cos(2*math.pi*q))

# Ackley potential function, nontrivial behaviour for q in (-2a,2a)
def ackley(q,**kwargs):
    a = kwargs['a'] if kwargs.get('a') else 20
    b = kwargs['b'] if kwargs.get('b') else 0.2
    c = kwargs['c'] if kwargs.get('c') else 2*math.pi
    n = q.shape[0]*q.shape[1]
    return -a*torch.exp(-b*torch.sqrt(torch.sum(q*q)/n))-torch.exp(torch.sum(torch.cos(c*q))/n) + a + torch.exp(torch.tensor(1.))

# isotropic and homogenous random potential function with gaussian correlation E[X(0)*X(tau)] ~ exp(-tau^2)
# to use first compute parameters through gauss_random_params() and pass
def gauss_random(q,params):
    N = q.shape[0]*q.shape[1]
    kapparange = params['kapparange'] if params.get('kapparange') else 0
    kappas = itertools.product(kapparange,repeat=N)
    S_vec = params['S'] if params.get('S') else 0
    Phi_vec = params['Phi'] if params.get('Phi') else 0
    Domega = params['Domega'] if params.get('Domega') else 0

    sum = 0;
    for kappa_vec in kappas:
        sum+=S_vec[kappa_vec]*torch.cos(torch.dot(torch.t(q)[0],omega(Domega,kappa_vec))+Phi_vec[kappa_vec]);
    return sum

def gauss_random_params(Kmax,Domega,N):
    S_vec = {}
    Phi_vec = {}
    kapparange = [i for i in range(-Kmax,Kmax+1)]
    kappas = itertools.product(kapparange,repeat=N)
    for kappa_vec in kappas:
        S_vec[kappa_vec] = torch.normal(0,torch.sqrt(2*Domega**N*S(omega(Domega,kappa_vec))));
        Phi_vec[kappa_vec] = 2*math.pi*torch.rand(1);

    return {'kapparange': kapparange,'S': S_vec,'Phi': Phi_vec,'Domega': Domega}

# auxilary function
def omega(Domega,args):
    n = len(args)
    om = []
    for arg in args:
        om+=[Domega/2*(2*arg-1)]
    #print(om)
    return torch.tensor(om)

# auxilary function
def S(om):
    n = om.shape[0]
    return torch.exp(-n*torch.dot(om,om)/2)*(n**(n/2+1))

# parabolic confinement
def parabolic(q,mu):
    n=q.shape[0]*q.shape[1];
    return 1/2*mu*torch.dot(torch.t(q)[0],torch.t(q)[0]);

# main function using Gentlest Ascent Dynamics
def find_saddle(*args,**kwargs):
    """
    Function exploring N-dimensional potentials and finding local saddles of index k using Gentlest Ascent Dynamics (GAD)

    OPTIONAL args:
    potential:  function            (default rastrigin) potential function with argument torch.Tensor of size [N,1] and returning scalar torch.Tensor; provided benchmark functions: rastrigin, ackley
    N:          int                 (default 2) dimension of potential function
    k:          0<=int<=N           (default 0) Morse index of saddle point (number of negative directions, for k=0 algorithm searches for local minima)
    alpha:      float>0             (default 0.001) single step size
    steps:      int                 (default 1000) number of steps in 'step' method
    epsilon:    float>0             (default 0.00001) threshold in 'conv' method
    method:     {'step','conv'}     (default 'conv') a) 'step' for method with fixed number of steps 'step', b) 'conv' for method based on convergence with 'epsilon' threshold
    trajectory: bool                (default False) returns whole q-trajectory if True
    q0:         N-list              (default random) initial search point
    verbose:    bool                (default False) verbose mode
    """
    conf = {}

    # define parameters
    # function dimension
    conf['N'] = kwargs['N'] if kwargs.get('N') else 2
    # saddle points' Morse index (number of unstable directions, varies between k=0...N)
    #k=0 probe minima whereas k=N maxima)
    conf['k'] = kwargs['k'] if kwargs.get('k') else 0
    # step size
    conf['alpha'] = kwargs['alpha'] if kwargs.get('alpha') else 0.001
    # number of steps if conf['method'] == 'steps'
    conf['steps'] = kwargs['steps'] if kwargs.get('steps') else 1000
    # update threshold below which algorithm stops if conf['method'] = 'conv'
    conf['epsilon'] = kwargs['epsilon'] if kwargs.get('epsilon') else 0.00001
    # choose method - fixed number of steps or epsilon convergence
    conf['method'] = kwargs['method'] if kwargs.get('method') else 'conv'
    # return whole trajectory
    conf['trajectory'] = True if kwargs.get('trajectory') else False
    # verbose mode
    conf['verbose'] = True if kwargs.get('verbose') else False

    # define initial position q0
    qdefault = torch.rand((conf['N'],1))
    if kwargs.get('q0'):
        temp = kwargs['q0']
        if type(temp)==list and len(temp)==conf['N']:
            q = torch.t(torch.tensor([temp]))
        else:
            q = qdefault
    else:
        q = qdefault

    #print(q.shape)

    # define potential function
    if kwargs.get('potential'): pfunc = kwargs['potential']
    # set Rastrigin function as default benchmark
    else: pfunc = rastrigin

    if conf['verbose']: print('configuration:',conf); print('initial argument:',np.transpose(q.numpy()));

    #declare shorthand variables
    N = conf['N']
    k = conf['k']
    steps = conf['steps']
    alpha = conf['alpha']
    epsilon = conf['epsilon']

    # run epsilon convergent algorithm
    if conf['method']=='conv':

        q.requires_grad_(True)
        V=rand_orthogonal(N,k)

        if conf['trajectory']: tr = torch.empty((N,1));
        D = 1
        j = 0
        while D > epsilon and j<100:
            j+=1
            q_grad = torch.autograd.grad(pfunc(q),q,create_graph=True)[0]
            q_grad.requires_grad_(True)
            hess_size = q_grad.shape[0]

            temp = [torch.autograd.grad(q_grad[i],q,retain_graph=True)[0] for i in range(hess_size)]
            q_hess = torch.cat(tuple(temp),0).reshape(hess_size,hess_size)

            with torch.no_grad():
                if k!=0:
                    VVt = torch.mm(V,torch.t(V))
                    qHessV = torch.mm(q_hess,V)
                    deltaq = torch.mm(torch.diag(torch.ones(N))-2*VVt,q_grad)

                    q = q - alpha*deltaq
                    V = V - alpha*torch.mm(torch.diag(torch.ones(N))-VVt,qHessV)
                else:
                    deltaq = q_grad
                    q = q - alpha*deltaq

                D = torch.norm(deltaq)
            if conf['verbose']: print('step',j,': norm(deltaq)=',D.detach().numpy());
            if conf['trajectory']: tr = torch.cat((tr,q),axis=1);
            q.requires_grad_(True)


        #if conf['trajectory']: return np.transpose(tr)[1:]
        if conf['trajectory']: return tr[:,1:].detach()
        return q.detach()

    # run fixed steps algorithm
    elif conf['method']=='step':

        q.requires_grad_(True)
        V=rand_orthogonal(N,k)

        if conf['trajectory']: tr = torch.empty((N,1));

        for j in range(steps):

            q_grad = torch.autograd.grad(pfunc(q),q,create_graph=True)[0]
            q_grad.requires_grad_(True)
            hess_size = q_grad.shape[0]

            temp = [torch.autograd.grad(q_grad[i],q,retain_graph=True)[0] for i in range(hess_size)]
            q_hess = torch.cat(tuple(temp),0).reshape(hess_size,hess_size)

            with torch.no_grad():
                if k!=0:
                    VVt = torch.mm(V,torch.t(V))
                    qHessV = torch.mm(q_hess,V)
                    q = q - alpha*torch.mm(torch.diag(torch.ones(N))-2*VVt,q_grad)
                    V = V - alpha*torch.mm(torch.diag(torch.ones(N))-VVt,qHessV)
                else:
                    q = q - alpha*q_grad

            if conf['verbose']: print('step',j);
            if conf['trajectory']: tr = torch.cat((tr,q),axis=1);
            q.requires_grad_(True)

        if conf['trajectory']: return tr[:,1:].detach()
        return q.detach()

# show potential function contour plot and GAD algorithm trajectory
#works only in N=2
def show_trajectory(tr,potential,*args,**kwargs):
    """
    Plotting function presenting trajectory found by find_saddle() when trajectory=True and N=2

    MANDATORY args:
    tr:         torch.Tensor        pytorch array returned by find_saddle(trajectory=True)
    potential:  function            potential function for trajectory tr

    OPTIONAL args:
    contours:   int                 (default 10) number of contours in potential function plot
    region:     {a,b}               (default {-1,1}) rectangular region size (a,b)x(a,b) centered at the saddle point
    resolution: int                 (default 200) resolution**2 is the number of all evaluated points inside rectangular region (a,b)x(a,b)
    """
    conf = {}

    print('Preparing contour plot...')
    # define parameters
    # trajectory
    traj = np.transpose(tr.numpy())
    pfunc = potential

    # define parameters
    # number of contours
    conf['contours'] = kwargs['contours'] if kwargs.get('contours') else 10
    # rectangular region size (a,b)x(a,b) centered at the saddle point
    conf['region'] = kwargs['region'] if kwargs.get('region') else {-1,1}
    # contour resolution (number of points inside rectangular region)
    conf['resolution'] = kwargs['resolution'] if kwargs.get('resolution') else 200

    pts=np.linspace(*conf['region'],num=conf['resolution'])
    x0,y0 = traj[-1]

    Z = np.empty((len(pts),len(pts)))
    X = x0+pts
    Y = y0+pts
    for i in range(len(pts)):
        for j in range(len(pts)):
            Z[i][j] = pfunc(torch.tensor([[x0+pts[i]],[y0+pts[j]]]))


    plt.axes().set_aspect('equal')
    plt.contourf(X,Y,Z,conf['contours'])
    plt.plot(np.transpose(traj)[0],np.transpose(traj)[1],'.-',color='red')
    plt.plot(x0,y0,'.',color='black')
    plt.show()
