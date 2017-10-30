# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 09:18:07 2017

@author: fmoret
"""

import gurobipy as gb
import numpy as np
import sys
from math import sqrt
import pandas as pd

from ADMM_pros_geo2Z import ADMM_Pros

# Class which can have attributes set
class expando(object):
    pass


class ADMM_Master:
    def __init__(self, b1, c1, Pmin, Pmax, PV, b2, c2, Load, Flex_load, tau, el_price_e, o, e, time, window, case, threshold, gamma_geo):

        self.data = expando()
        self.variables = expando()

        self.results = expando()
        self.params = expando()
        
        self.data.gen_lin_cost = b1
        self.data.gen_quad_cost = c1
        self.data.gen_lb = Pmin
        self.data.gen_ub = Pmax
        self.data.PV = PV
        self.data.cons_lin_cost = b2
        self.data.cons_quad_cost = c2
        self.data.cons_ub = - Load
        self.data.cons_lb = - (Load + 2*Flex_load)
        self.data.goal = - (Load + Flex_load)
        self.data.tau = tau
        self.data.el_price_e = el_price_e
        self.data.rho = o
        self.data.tol = e
        self.data.gamma_geo = gamma_geo
        self.data.A = np.array([[1,1,0],[-1,0,1],[0,-1,-1]])
        
        self.params.num_pros = b2.shape[1]
        self.case = case
        self.window = window
        self.params.time = time
        
        self.variables.p_k = np.zeros((time, self.params.num_pros))
        self.variables.l_k = np.zeros((time, self.params.num_pros))
        self.variables.q_k = np.zeros((time, self.params.num_pros))
        self.variables.q_pos_k = np.zeros((time, self.params.num_pros))
        self.variables.alfa_k = np.zeros((time, self.params.num_pros))
        self.variables.beta_k = np.zeros((time, self.params.num_pros))
        self.variables.pow_imp = np.zeros(time)
        self.variables.pow_exp = np.zeros(time)
        self.variables.pow_comm = np.zeros((time,3))
        self.variables.price_comm = np.zeros((time,3))
        self.variables.price_IE = np.zeros((time,2))
        
        self.imbalance = np.zeros((time+1, self.params.num_pros))
        self.iter_max = 10000
        self.weight = 3/self.params.num_pros
        self.threshold = threshold
        
        self._init_model()
        self._init_subproblems()
        self.optimize()
        #self.clean()

    ###
    # Model Building
    ###
    def _init_model(self):
        self.model = gb.Model()
        self.model.setParam('OutputFlag', False)
        self.model.update()
        self.stdold = sys.stdout
#        sys.stdout = open('pros_out.txt','w')

    def _build_variables(self):
        m = self.model
        self.variables.q_imp = np.array([m.addVar() for k in range(24)])
        self.variables.q_exp = np.array([m.addVar() for k in range(24)])
        self.variables.q_comm = np.array([m.addVar() for i in range(3) for k in range(24)])
        self.variables.q_cpp = np.array([m.addVar() for i in range(3) for k in range(24)])
        self.variables.q_comm = np.transpose(self.variables.q_comm.reshape(3,24))
        self.variables.q_cpp = np.transpose(self.variables.q_cpp.reshape(3,24))
        for k in range(24):
            for i in range(3):
                m.addConstr(self.variables.q_comm[k,i] <= self.variables.q_cpp[k,i])
                m.addConstr(self.variables.q_comm[k,i] >= -self.variables.q_cpp[k,i])
        m.update()

    def _build_objective(self, t):
        m = self.model
        q = self.variables.q_k[self.temp,:]
        alfa = self.variables.alfa_k[self.temp,:]
        beta = self.variables.beta_k[self.temp,:]
        pr_exp = -self.data.el_price_e[self.temp]
        pr_imp = self.data.el_price_e[self.temp] + 0.1# + self.data.tau
        price_0 = self.variables.price_comm[self.temp,:]
        price_1 = self.variables.price_IE[self.temp,:]
        gamma_geo = self.data.gamma_geo 
        o = self.data.sigma
        res01 = self.weight*(q[:,range(0,5)].sum(axis=1)-self.variables.q_comm[:,0]-self.variables.q_comm[:,1])
        res02 = self.weight*(q[:,range(5,10)].sum(axis=1)+self.variables.q_comm[:,0]-self.variables.q_comm[:,2])
        res03 = self.weight*(q[:,range(10,15)].sum(axis=1)+self.variables.q_comm[:,1]+self.variables.q_comm[:,2])
        res1 = alfa.sum(axis=1) - self.variables.q_imp
        res2 = beta.sum(axis=1) - self.variables.q_exp
        
        self.objective = (sum(pr_imp*self.variables.q_imp + pr_exp*self.variables.q_exp) + sum(self.variables.q_cpp[:,0]*gamma_geo[0]+self.variables.q_cpp[:,1]*gamma_geo[1]+self.variables.q_cpp[:,2]*gamma_geo[2])
                          + sum(price_0[:,0]*res01+price_0[:,1]*res02+price_0[:,2]*res03) + sum(price_1[:,0]*res1+price_1[:,1]*res2) + o/2*(sum(res01*res01+res02*res02+res03*res03)+sum(res1*res1+res2*res2)))
        m.setObjective(self.objective)
        m.update()

    def _init_subproblems(self):
        self._init_prosumers()

    def _init_prosumers(self):
        # Only build submodels if they don't exist or a rebuild is forced.
        if not hasattr(self, 'submodels_pros'):
            self.submodels_pros = {i: ADMM_Pros(self, idx=i) for i in range(self.params.num_pros)}
                                   
    def _model_update(self, t):
        self._build_variables()
        self._build_objective(t)
        self._pros_update(t)
                                  
    def _pros_update(self, t):
        for i in range(self.params.num_pros):
            self.submodels_pros.get(i)._update_model_(t)
            
    def optimize(self):
        resul = []
        resul2 = []
        fl = []

        for t in np.arange(0,self.params.time,24): #8,self.params.time
            self.temp = range(t,t+24) 
            self.variables.iter = 0
            self.variables.r_k = np.ones(5*24)#+self.params.num_pros)
            real_rk = np.ones(5*24)
            self.variables.s_k = 1.0
            self.data.sigma = self.data.rho
            res = []
            res2 = []
            res3 = []
            res4 = []
            res5 = []
            res6 = []
            res7 = []
            res8 = []
            res9 = []
            res10 = []
            self._model_update(t)
            flag = 0
            stack = 1000*np.ones(100)
            
            while (np.linalg.norm(real_rk)>self.data.tol or self.variables.s_k>self.data.tol) and self.variables.iter<self.iter_max and np.linalg.norm(real_rk)<10**10: #np.linalg.norm
                self.variables.iter = self.variables.iter +1
                
                #Solve subproblems
                for j in range(self.params.num_pros):
                    sol = self.submodels_pros[j].optimize()
                    self.variables.p_k[self.temp,j] = np.copy(sol[:,0]) 
                    self.variables.l_k[self.temp,j] = np.copy(sol[:,1]) 
                    self.variables.q_k[self.temp,j] = np.copy(sol[:,2]) 
                    self.variables.alfa_k[self.temp,j] = np.copy(sol[:,3]) 
                    self.variables.beta_k[self.temp,j] = np.copy(sol[:,4])
                    pp = np.copy(sol[:,5])

                #Solve main problem
                self._build_objective(t)
                self.model.optimize()
                imp_old = self.variables.pow_imp[self.temp]
                exp_old = self.variables.pow_exp[self.temp]
                q_comm_old = np.copy(self.variables.pow_comm[self.temp,:])
                for k in range(24):
                    self.variables.pow_imp[t+k] = self.variables.q_imp[k].x
                    self.variables.pow_exp[t+k] = self.variables.q_exp[k].x
                    for j in range(3):
                        self.variables.pow_comm[t+k,j] = self.variables.q_comm[k,j].x

                #Calculate residulas
#                print(self.variables.q_k[self.temp,:][:,range(0,5)])
#                print(self.variables.q_k[self.temp,range(0,5)].sum(axis=1))
                self.variables.r_k[range(0,24)] = self.weight*(self.variables.q_k[self.temp,:][:,range(0,5)].sum(axis=1)-self.variables.pow_comm[self.temp,0]-self.variables.pow_comm[self.temp,1])
                self.variables.r_k[range(24,48)] = self.weight*(self.variables.q_k[self.temp,:][:,range(5,10)].sum(axis=1)+self.variables.pow_comm[self.temp,0]-self.variables.pow_comm[self.temp,2])
                self.variables.r_k[range(48,72)] = self.weight*(self.variables.q_k[self.temp,:][:,range(10,15)].sum(axis=1)+self.variables.pow_comm[self.temp,1]+self.variables.pow_comm[self.temp,2])
                self.variables.r_k[range(72,96)] = (self.variables.alfa_k[self.temp,:].sum(axis=1) - self.variables.pow_imp[self.temp])
                self.variables.r_k[range(96,120)] = self.variables.beta_k[self.temp,:].sum(axis=1) - self.variables.pow_exp[self.temp]

                if max(abs(np.linalg.norm(self.variables.r_k)*np.ones(100)-stack))<0.01*np.linalg.norm(self.variables.r_k) and self.data.sigma<0.5:
                    self.data.sigma = self.data.sigma*2
                    stack = 100*np.ones(100)
                elif np.linalg.norm(self.variables.r_k)-np.mean(stack)>0.1*np.linalg.norm(self.variables.r_k):
                    self.data.sigma = self.data.sigma/3
                    stack = 100*np.ones(100)
                stack[range(1,100)] = np.copy(stack[range(0,99)])
                stack[0] = np.linalg.norm(self.variables.r_k)

                Dcomm1 = (self.variables.pow_comm[self.temp,0] - q_comm_old[:,0])+(self.variables.pow_comm[self.temp,1] - q_comm_old[:,1])
                Dcomm2 = (self.variables.pow_comm[self.temp,0] - q_comm_old[:,0])-(self.variables.pow_comm[self.temp,2] - q_comm_old[:,2])
                Dcomm3 = (self.variables.pow_comm[self.temp,2] - q_comm_old[:,2])+(self.variables.pow_comm[self.temp,1] - q_comm_old[:,1])
                Dimp = self.variables.pow_imp[self.temp]-imp_old
                Dexp = self.variables.pow_exp[self.temp]-exp_old
                self.variables.s_k = sqrt(self.params.num_pros)*self.data.sigma*sqrt(sum(Dimp*Dimp + Dexp*Dexp + Dcomm1*Dcomm1 + Dcomm2*Dcomm2 + Dcomm3*Dcomm3))

                real_rk = np.copy(self.variables.r_k)
                real_rk[range(0,72)] = real_rk[range(0,24)]/self.weight
                
                #Update price
                self.variables.price_comm[self.temp,0] = self.variables.price_comm[self.temp,0] + self.data.sigma*self.variables.r_k[range(0,24)]
                self.variables.price_comm[self.temp,1] = self.variables.price_comm[self.temp,1] + self.data.sigma*self.variables.r_k[range(24,48)]
                self.variables.price_comm[self.temp,2] = self.variables.price_comm[self.temp,2] + self.data.sigma*self.variables.r_k[range(48,72)]
                self.variables.price_IE[self.temp,0] = self.variables.price_IE[self.temp,0] + self.data.sigma*self.variables.r_k[range(72,96)]
                self.variables.price_IE[self.temp,1] = self.variables.price_IE[self.temp,1] + self.data.sigma*self.variables.r_k[range(96,120)]

#                print(t, '-',self.variables.iter,'     Primal Residual ', np.linalg.norm(real_rk)/24, '     Dual Residual ', self.variables.s_k, '     Sigma ', self.data.sigma)

                if np.linalg.norm(real_rk)>10**10:
                    flag = 2
                if self.variables.iter==self.iter_max:
                    flag = 1
                
#                for k in range(24):
#                    d = {'imp': self.variables.pow_imp[t+k],
#                     'exp': self.variables.pow_exp[t+k],
#                     's_k': self.variables.s_k,
#                     'r_k': np.linalg.norm(self.variables.r_k),
#                     'r_k[0]': real_rk[k],
#                     'r_k[1]': real_rk[24+k],
#                     'r_k[2]': real_rk[48+k],
#                     'r_k[3]': real_rk[72+k],
#                     'r_k[4]': real_rk[96+k]
#                     }
#                    aa = {'alfa[%s]'%(i): self.variables.alfa_k[t+k,i] for i in range(self.params.num_pros)}
#                    bb = {'q_12' : self.variables.pow_comm[t+k,0],
#                          'q_13' : self.variables.pow_comm[t+k,1],
#                          'q_23' : self.variables.pow_comm[t+k,2]}
#                    cc = {'load[%s]'%(i): self.variables.l_k[t+k,i] for i in range(self.params.num_pros)}
#                    dd = {'pow[%s]'%(i): self.variables.p_k[t+k,i] for i in range(self.params.num_pros)}
#                    ee = {'price[%s]'%(i): pp[i] for i in range(self.params.num_pros)}
#                    ff = {'price_imp': self.variables.price_IE[t+k,0],
#                          'price_exp': self.variables.price_IE[t+k,1],
#                          'price_comm[0]': self.variables.price_comm[t+k,0],
#                          'price_comm[1]': self.variables.price_comm[t+k,1],
#                          'price_comm[2]': self.variables.price_comm[t+k,2]}
#                    gg = {'q[%s]'%(i): self.variables.q_k[t+k,i] for i in range(self.params.num_pros)}
#                    ii = {'beta_k[%s]'%(i): self.variables.beta_k[t+k,i] for i in range(self.params.num_pros)}
#                    res.append(d)
#                    res2.append(aa)
#                    res3.append(bb)
#                    res4.append(cc)
#                    res5.append(dd)
#                    res6.append(ee)
#                    res7.append(ff)
#                    res8.append(gg)
#                    res10.append(ii)

            print('Geo 3 Zones - Timestamp %g' % (t), '  Flag ', flag)
                
            fl.append(flag)
#            res = pd.DataFrame(res)
#            res2 = pd.DataFrame(res2)
#            res3 = pd.DataFrame(res3)
#            res4 = pd.DataFrame(res4)
#            res5 = pd.DataFrame(res5)
#            res6 = pd.DataFrame(res6)
#            res7 = pd.DataFrame(res7)
#            res8 = pd.DataFrame(res8)
#            res9 = pd.DataFrame(res9)
#            res10 = pd.DataFrame(res10)
#            resul.append(pd.concat([res,res3,res2,res10,res4,res5,res8], axis=1))
#            resul2.append(pd.concat([res7,res6], axis=1))
#            
#        self.results = resul
#        self.prices = resul2
        self.flag = fl

    def clean(self):
        sys.stdout = self.stdold
        del self.model
        [self.submodels_pros.get(sm).clean() for sm in self.submodels_pros.keys()]
         