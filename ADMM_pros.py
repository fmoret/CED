# Import Gurobi Library
import gurobipy as gb
import numpy as np
#from math import log10

# Class which can have attributes set.
class expando(object):
    pass

# Optimization class
class ADMM_Pros:
    def __init__(self, MP, idx):
        self.variables = expando()
        self.params = expando()
        
        self.MP = MP
        self.params.idx = idx
        self.Pmin = self.MP.data.gen_lb[idx]   
        self.Pmax = self.MP.data.gen_ub[idx]  
        self.PV = self.MP.data.PV[:,idx] 
        self.goal = self.MP.data.goal[:,idx] 
        b2 = self.MP.data.cons_lin_cost[:,idx]
        c2 = self.MP.data.cons_quad_cost[:,idx]
        b1 = self.MP.data.gen_lin_cost[:,idx]
        c1 = self.MP.data.gen_quad_cost[:,idx]
        self.Lmax = self.MP.data.cons_ub[:,idx]
        self.Lmin = self.MP.data.cons_lb[:,idx]
        self.y0_c = 0.01 + b2 - c2*(self.Lmax+self.Lmin)/(self.Lmax-self.Lmin)
        self.mm_c = 2*c2/(self.Lmax-self.Lmin)
        if sum(abs(self.mm_c[np.isinf(self.mm_c)]))>0:
            self.y0_c[np.isinf(abs(self.mm_c))] = 0.01 + b2[np.isinf(abs(self.mm_c))]
            self.mm_c[np.isinf(abs(self.mm_c))] = 0

        self.y0_g = 0.01 + b1 - c1*(self.Pmax+self.Pmin)/(self.Pmax-self.Pmin)
        self.mm_g = 2*c1/(self.Pmax-self.Pmin)
        if sum(abs(self.mm_g[np.isinf(self.mm_g)]))>0:
            self.y0_g[np.isinf(abs(self.mm_g))] = 0.01 + b1[np.isinf(abs(self.mm_g))]
            self.mm_g[np.isinf(abs(self.mm_g))] = 0
        self.y0_g[np.isnan(self.y0_g)] = 0
        self.mm_g[np.isnan(self.mm_g)] = 0

        self._build_model_()        

    def optimize(self):
        self._build_objective_()
        self.model.optimize()
        x = np.random.normal(0,1)
        if x>self.MP.threshold:
            for k in range(24):
                self.beta_old[k] = self.variables.beta[k].x
                self.alfa_old[k] =self.variables.alfa[k].x  
                self.q_old[k] =self.variables.q[k].x  
                self.pow_old[k] = self.variables.power[k].x
                self.load_old[k] = self.variables.load[k].x
                self.price_old[k] = self.model.getConstrByName("pow_bal[%s]"%k).Pi
        res = np.empty((24,6))
        res[:,0] = self.pow_old
        res[:,1] = self.load_old
        res[:,2] = self.q_old
        res[:,3] = self.alfa_old
        res[:,4] = self.beta_old
        res[:,5] = self.price_old 
        return res
        
    ###
    #   Model Building
    ###
    def _build_model_(self):
        self.model = gb.Model()
        self.model.setParam('OutputFlag', False)
        self.model.update()
        
    def _update_model_(self, t):
        self.t = t
        self.beta_old = np.zeros(24)
        self.alfa_old = np.zeros(24)
        self.q_old = np.zeros(24)
        self.pow_old = np.zeros(24)
        self.load_old = np.zeros(24)
        self.price_old = np.zeros(24)
        self._build_variables_()
        self._build_constraint_()
        self._build_objective_()

    def _build_variables_(self,):
        m = self.model
        self.variables.load = np.array([m.addVar(lb=self.Lmin[self.t+k], ub=self.Lmax[self.t+k]) for k in range(24)])
        self.variables.power = np.array([m.addVar(lb=self.Pmin+self.PV[self.t+k], ub=self.Pmax+self.PV[self.t+k]) for k in range(24)])
        self.variables.beta = np.array([m.addVar() for k in range(24)])
        self.variables.alfa = np.array([m.addVar() for k in range(24)])
        self.variables.q = np.array([m.addVar(-gb.GRB.INFINITY) for k in range(24)])
        self.variables.q_pos = np.array([m.addVar() for k in range(24)])
        m.update()
        
    def _build_constraint_(self,):
        m = self.model
        for k in range(24):
            m.addConstr(self.variables.load[k] + self.variables.power[k] + self.variables.q[k] + self.variables.alfa[k] - self.variables.beta[k] == 0, name="pow_bal[%s]"%k) 
            m.addConstr(self.variables.q[k] <= self.variables.q_pos[k])
            m.addConstr(self.variables.q[k] >= -self.variables.q_pos[k])
#        t = self.MP.temp[0]
#        for j in np.arange(0,24,self.window):
#            m.addConstr(sum(self.goal[range(t+j,t+j+self.window)] - self.variables.load[range(t+j,t+j+self.window)]) == 0)
        m.addConstr(sum(self.goal[self.MP.temp] - self.variables.load) == 0)
        m.update()

    def _build_objective_(self):
        m = self.model
        temp = self.MP.temp
        y0_c = self.y0_c[temp]
        mm_c = self.mm_c[temp]
        y0_g = self.y0_g[temp]
        mm_g = self.mm_g[temp]
        price0 = self.MP.variables.price_comm[temp]
        price1 = self.MP.variables.price_IE[temp,:]
        o = self.MP.data.sigma
        res0 = self.MP.variables.r_k[range(0,24)] + self.MP.weight*(self.variables.q - self.q_old)
        res1 = self.MP.variables.r_k[range(24,48)] + (self.variables.alfa - self.alfa_old)
        res2 = self.MP.variables.r_k[range(48,72)] + self.variables.beta - self.beta_old
        self.objective = (sum(y0_c*(self.variables.load) + mm_c/2*(self.variables.load*self.variables.load) + y0_g*(self.variables.power) + mm_g/2*(self.variables.power*self.variables.power))
                          + sum(0.001*self.variables.q_pos) + sum(price0*res0) + sum(price1[:,0]*res1+price1[:,1]*res2) + o/2*(sum(res0*res0)+sum(res1*res1+res2*res2))) #+ sum(0.1*self.variables.alfa + 0.05*self.variables.beta 
        m.setObjective(self.objective)
        m.update()
        
    def clean(self):
        del self.model