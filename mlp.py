import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
from time import time
from multiprocessing import  Pool
from scipy.optimize import linear_sum_assignment


class P2PEngine(object):
    def __init__(self, env, config, pure_bandit,predict_mask):
        self.config = config
        self.pure_bandit = pure_bandit
        self.predict_mask = predict_mask
        self.predictor = LinearRegression(fit_intercept=False, n_jobs=-1)
        self.mask_predictor = LinearRegression()
        
        self.n = config['points_per_iter']
        self.m = config['feature_dim']
        self.d = config['label_dim']
        self.d_max = config['label_max']
        self.delta = config['delta']
        self.w = np.zeros((self.n, self.d))
        self.A = np.tile(np.eye(self.d), (self.n, 1, 1))
        self.mu_hat = np.zeros((self.n, self.d))
        self.rw = np.zeros((self.n, self.d))
        self.beta = max(128 * self.d * np.log(1) * np.log(1/self.delta), np.square(8/3 * np.log(1/self.delta)))
        self.mu = env.mu
        self.w_known = np.zeros((self.n, self.d))
        self.volunteer_info = env.r

    def p2p_an_epoch(self, data_loader, test_feature, epoch_id):
        """For each epoch, learn then optimize to figure out what the optimal value of w is
        
        Arguments: 
            data_loader: PyTorch data loader containing each data point
            test_feature: Value of x we receive; we use this to predict w 
            epoch_id: Which epoch we're on; used to index into data_loader 

        Returns:
            Predicted optimal w, and the amount of time for computation
        """
        
        self.beta = max(128 * self.d * np.log(epoch_id) * np.log(epoch_id*epoch_id/self.delta),
        np.square(8/3 * np.log(epoch_id*epoch_id/self.delta)))
        start_time = time()
        if not self.pure_bandit:
            self.learn(data_loader)
        if self.predict_mask:
            self.learn_mask(data_loader)
        self.optimize(test_feature)
        end_time = time()
        return self.w, end_time - start_time

    def learn(self, data_loader):
        """Add the pair (x,c) to our predictor function, which is currently linear regression
        
        Arguments:
            data_loader: PyTorch data loader with our data (x and c)

        Returns: Nothing
        
        Side Effects: Re-fits our predictor function with all of the data_loader data points
        """

        self.predictor.fit(data_loader.dataset.feature, data_loader.dataset.label)
        
    def format_mask_data(self,x,r):
        """Turn a matrix x (containing task data) and r, containing volunteer info, into
            A combined matrix, where each row of x is x[i] + r
            
        Arguments:
            x: A numpy array, of size n x m 
            r: A numpy array of size d x n_attributes

        Returns: Numpy array of size n x (m+d*n_attributes)
        """
        
        x = x[:,self.m:]
        r = r.flatten()
        r_formatted = np.stack((r,) * len(x), axis=0)
        combined_array = np.concatenate([x,r_formatted],axis=1)
        return combined_array
        
    def learn_mask(self, data_loader):
        """Add the pair ((x,y,),M) to our predictor function, which is currently linear regression
        
        Arguments:
            data_loader: PyTorch data loader with our data (x)
            volunteer_info: The numpy array y
            true_mask: The true value of the mask, M
            
        Returns: Nothing
        
        Side Effects: Re-fits our mask_predictor function with ((x,y),M)
        """
        
        x = data_loader.dataset.feature
        formatted_data = self.format_mask_data(x,self.volunteer_info)
                        
        self.mask_predictor.fit(formatted_data,data_loader.dataset.mask)

    def optimize(self, test_feature):
        """Run an optimization function for each data point in the epoch (n of them) 
            Run this in parallel
            
        Arguments: Test feature; matrix of what x is
        
        Return: Nothing
        
        Side Effects: Computes self.w by calling helper_optimize for each batch
        """
        
        n_cores = 4
        lst = np.arange(self.n)
        n_lst = int(len(lst)/n_cores)
        test_features_split = [test_feature[i:i + n_lst, :] for i in range(0, len(lst), n_lst)]
        pool = Pool(len(test_features_split))
        w_list = pool.map(self.helper_optimize, test_features_split)
        pool.close()
        pool.join()
        self.w = np.concatenate(w_list, axis=0)

    def helper_optimize(self, test_feature):
        """Given some value of x, optimize for w by doing the following: 
            1. Predict what c is using the predictor function (if applicable)
            2. Run a double optimization for w and nu (which is our predicted value of mu; 
                the hidden part of the loss function
                We have a constraint on nu (must be close to mu_hat) and w (must be 0-1 vector)
        
        Arguments: Test Feature: Some value of x
        
        Returns: Our predicted w for this x
        """
        
        solver = pyomo.opt.SolverFactory('ipopt')
        solver.options['print_level'] = 5
        solver.options['max_cpu_time'] = int(100)
        solver.options['warm_start_init_point'] = 'yes'
        
        w = -np.inf * np.ones((test_feature.shape[0], self.d))
        
        for i in range(test_feature.shape[0]):
            x = test_feature[i,:]
            if self.pure_bandit:
                c_hat = np.zeros(self.d)
            else:
                c_hat = self.predictor.predict(x.reshape(1, -1)).squeeze()
                
            if self.predict_mask:
                combined_data = self.format_mask_data(x.reshape(1,-1),self.volunteer_info)
                m_hat = self.mask_predictor.predict(combined_data).squeeze()
            else:
                m_hat = np.ones(self.d)

            model = ConcreteModel()
            model.dSet = Set(initialize=range(self.d))
            model.w = Var(model.dSet,domain=Binary)
            model.nu = Var(model.dSet)
            for j in range(self.d):
                model.w[j].value = (2*np.random.rand() - 1)/np.sqrt(self.d)
                model.nu[j].value = (2*np.random.rand() - 1)/np.sqrt(self.d)

            model.w_constraint = Constraint(expr=sum(model.w[j] for j in range(self.d)) <= self.d_max)  
            expr1 = sum(self.A[i,j,k] * (model.nu[j] - self.mu_hat[i,j]) *
            (model.nu[k] - self.mu_hat[i,k]) for j in range(self.d) for k in range(self.d))
            model.nu_constraint = Constraint(expr= expr1 <= self.beta ** .5)
            
            model.obj = Objective(expr=sum((-c_hat[j] + model.nu[j]) * model.w[j] * m_hat[j] for j in range(self.d)), sense=minimize)

            try:
                result = solver.solve(model, tee = False, keepfiles = False)
            except ValueError:
                print("Value error!!")
                w[i,:] = -self.mu_hat[i,:]/np.linalg.norm(self.mu_hat[i,:], 2)
                continue

            if (result.solver.status == SolverStatus.ok) and (result.solver.termination_condition == TerminationCondition.optimal):
                self.feasible = True
                w[i,:] = [value(model.w[j]) for j in range(self.d)]
            else:
                self.feasible = False
                print('Encountered an attribute error')
        return w


    def update_bandit(self, ro, rb):
        """Update our regret function, (rw), and our generation functions (A, mu_hat)
        THis corresponds to lines 15 and 16 in Algorithm 2
        
        Arguments: 
            ro: Our regret function from optimizing (no mu) 
            rb: Our regret function from the bandit portion (with mu)

        Returns: Nothing
        
        Side Effects: Updates rw, A, and mu_hat
        """
        
        if self.pure_bandit:
            rb = ro + rb
        for i in range(self.n):
            self.rw[i,:] += rb[i] * self.w[i,:]
            self.A[i,:,:] = self.A[i,:,:] + np.outer(self.w[i,:], self.w[i,:])
            self.mu_hat[i,:] = np.matmul(np.linalg.inv(self.A[i,:,:]), self.rw[i,:])

    def p2p_known_mu(self, test_label, mask):
        """Given that we know c and mu, find the offline optimal solution
        Do this by minimizing (-c+mu)*w, with the constraint that |w| <= self.d
        
        Arguments: 
            test_label: n x d 0-1 matrix, with information on who's available
            mask: n x d 0-1 matrix, with information on who's available, based on volunteer properties

        Returns: 
            w_known: Optimal w given all the information
        """
        
        for i in range(self.n):
            lp = gp.Model("lp" + str(i))
            c = (test_label[i,:]*mask[i,:]).squeeze()
            w = lp.addMVar(shape=self.d, lb=-GRB.INFINITY, name="w", vtype=gp.GRB.BINARY)
            lp.setObjective((-c + self.mu) @ w, GRB.MINIMIZE)            
            lp.addConstr(gp.quicksum(w[i] for i in range(self.d)) <= self.d_max,name="norm")
            lp.Params.OutputFlag = 0
            try:
                lp.optimize()
                self.w_known[i,:] = w.X
            except gp.GurobiError as e:
                pass
            except AttributeError:
                print('Encountered an attribute error')
        return self.w_known


    def get_matches(self,action,labels,mask):
        """Given some actions and corresponding labels, find the optimal batch match
            The actions refere to which volunteers we alert; of these, those with label = 1 are available
            We additionally incorporate our mask information
            So using those we alert + are available, match them using the Hungarian Algorithm
        
        Arguments:
            Action: n x d 0-1 matrix, with information on who we're alerting 
            Labels: n x d 0-1 matrix, with information on who's available
            Mask: n x d 0-1 matrix, with information on who's available, using extra attributes
            
        Returns: List of tuples, which contain valid (trip,volunteer) matches
        """
        
        true_matches = action*labels*mask
        hungarian_algorithm = linear_sum_assignment(-true_matches)
        
        matches = list(zip(*hungarian_algorithm))
        valid_matches = [(i,j) for (i,j) in matches if true_matches[i][j]>0.5]
        return valid_matches 
