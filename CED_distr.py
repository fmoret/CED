# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:40:18 2016

@author: fmoret
"""
import os
import gurobipy as gb
import numpy as np
import pandas as pd
dd1 = '/zhome/81/d/95321/IEEE/Vers 3'
dd2 = '/zhome/81/d/95321/IEEE/Input Data 2'
os.chdir(dd1)

#%%
os.chdir(dd2)
import shelve
filename='input_data_2.out'
d = shelve.open(filename, 'r')
el_price = d['tot_el_price'].as_matrix()[:,5][0::2]/1000
b2 = d['b2']
c2 = d['c2']
b1 = d['b1']
c1 = d['c1']
Pmin = d['Pmin']
Pmax = d['Pmax']
PV = d['PV']
Load = d['Load']
Flex_load = d['Flex_load']
Agg_load = Load + Flex_load
PostCode = d['PostCode']

os.chdir(dd1)
n = b2.shape[1]
g = b1.shape[1]
TMST = b2.shape[0]

Lmin = - (Agg_load + Flex_load)
Lmax = - Load
el_price_e = el_price
tau = 0.1
window = 4

y0_c = 0.01 + b2 - c2*(Lmax+Lmin)/(Lmax-Lmin)
mm_c = 2*c2/(Lmax-Lmin)
if sum(abs(mm_c[np.isinf(mm_c)]))>0:
    y0_c[np.isinf(abs(mm_c))] = 0.01 + b2[np.isinf(abs(mm_c))]
    mm_c[np.isinf(abs(mm_c))] = 0

y0_g = 0.01 + b1 - c1*(Pmax+Pmin)/(Pmax-Pmin)
mm_g = 2*c1/(Pmax-Pmin)
if sum(abs(mm_g[np.isinf(mm_g)]))>0:
    y0_g[np.isinf(abs(mm_g))] = 0.01 + b1[np.isinf(abs(mm_g))]
    mm_g[np.isinf(abs(mm_g))] = 0
y0_g[np.isnan(y0_g)] = 0
mm_g[np.isnan(mm_g)] = 0
     
#%% Community Trading
CT_p_sol = np.zeros((g,TMST))
CT_l_sol = np.zeros((n,TMST))
CT_q_sol = np.zeros((n,TMST))
CT_alfa_sol = np.zeros((n,TMST))
CT_beta_sol = np.zeros((n,TMST))
CT_imp_sol = np.zeros(TMST)
CT_exp_sol = np.zeros(TMST)
CT_obj_sol = np.zeros(TMST)
CT_price_sol = np.zeros((n,TMST))
CT_price2_sol = np.zeros((3,TMST))

for t in np.arange(0,TMST,24):
    temp = range(t,t+24)
    # Create a new model
    CT_m = gb.Model("qp")
    CT_m.setParam( 'OutputFlag', False )
    # Create variables
    p = np.array([CT_m.addVar(lb=Pmin[i]+PV[t+k,i], ub=Pmax[i]+PV[t+k,i]) for i in range(n) for k in range(24)])
    l = np.array([CT_m.addVar(lb=Lmin[t+k,i], ub=Lmax[t+k,i]) for i in range(n) for k in range(24)])
    q_imp = np.array([CT_m.addVar() for k in range(24)])
    q_exp = np.array([CT_m.addVar() for k in range(24)])
    alfa = np.array([CT_m.addVar() for i in range(n) for k in range(24)])
    beta = np.array([CT_m.addVar() for i in range(n) for k in range(24)])
    q_pos = np.array([CT_m.addVar() for i in range(n) for k in range(24)])
    q = np.array([CT_m.addVar(lb = -gb.GRB.INFINITY) for i in range(n) for k in range(24)])
    gamma_i = (el_price_e[temp] + 0.1)
    gamma_e = -el_price_e[temp]
    CT_m.update()
    
    p = np.transpose(p.reshape(n,24))
    l = np.transpose(l.reshape(n,24))
    q = np.transpose(q.reshape(n,24))
    q_pos = np.transpose(q_pos.reshape(n,24))
    alfa = np.transpose(alfa.reshape(n,24))
    beta = np.transpose(beta.reshape(n,24))
    
    # Set objective: 
    obj = (sum(sum(y0_c[temp,:]*l + mm_c[temp,:]/2*l*l) + sum(y0_g[temp,:]*p + mm_g[temp,:]/2*p*p)) 
           + sum(gamma_i*q_imp + gamma_e*q_exp) + sum(0.001*sum(q_pos)))
    CT_m.setObjective(obj)
    # Add constraint
    for k in range(24):
        CT_m.addConstr(sum(q[k,:]) == 0, name="comm[%s]"%(k))
        for i in range(n):
            CT_m.addConstr(p[k,i] + l[k,i] + q[k,i] + alfa[k,i] - beta[k,i] == 0, name="pros[%s,%s]"% (k,i))
            CT_m.addConstr(q[k,i] <= q_pos[k,i])
            CT_m.addConstr(q[k,i] >= -q_pos[k,i])
        CT_m.addConstr(sum(alfa[k,:]) - q_imp[k] == 0, name="imp_bal[%s]"%(k))
        CT_m.addConstr(sum(beta[k,:]) - q_exp[k] == 0, name="exp_bal[%s]"%(k))
    for i in range(n):
        CT_m.addConstr(sum(Agg_load[temp,i] + l[:,i]) == 0)
#        for j in np.arange(0,24,window):
#            CT_m.addConstr(sum(Agg_load[range(t+j,t+j+window),i] + l[range(j,j+window),i]) == 0)
    CT_m.update()    
        
    CT_m.optimize()
    for k in range(24):
        for i in range(n):
            CT_price_sol[i,t+k] = CT_m.getConstrByName("pros[%s,%s]"%(k,i)).Pi
            CT_p_sol[i,t+k] = p[k,i].x
            CT_l_sol[i,t+k] = l[k,i].x
            CT_q_sol[i,t+k] = q[k,i].x
            CT_alfa_sol[i,t+k] = alfa[k,i].x
            CT_beta_sol[i,t+k] = beta[k,i].x
        CT_price2_sol[0,t+k] = CT_m.getConstrByName("comm[%s]"%k).Pi
        CT_price2_sol[1,t+k] = CT_m.getConstrByName("imp_bal[%s]"%k).Pi
        CT_price2_sol[2,t+k] = CT_m.getConstrByName("exp_bal[%s]"%k).Pi
        CT_imp_sol[t+k] = q_imp[k].x
        CT_exp_sol[t+k] = q_exp[k].x

    del CT_m

CT_IE_sol = CT_imp_sol - CT_exp_sol


#%% ADMM optimisation via master-sub classes
from ADMM_master import ADMM_Master
o = 0.01
e = 0.00001
asynch = -100
weight = 1/n

#% Cost of Import/Export not considered ==> case = 0
Mast = ADMM_Master(b1, c1, Pmin, Pmax, PV, b2, c2, Load, Flex_load, tau, el_price_e, o, e, TMST, window, 0, asynch)
#ADMM_summary = Mast.resu1lts
#ADMM_prices = Mast.prices
ADMM_flag = Mast.flag

#%%
max_l = abs(Mast.variables.l_k - np.transpose(CT_l_sol)).max()
max_p = abs(Mast.variables.p_k - np.transpose(CT_p_sol)).max()
max_q = abs(Mast.variables.q_k - np.transpose(CT_q_sol)).max()
max_alfa = abs(Mast.variables.alfa_k - np.transpose(CT_alfa_sol)).max()
max_beta = abs(Mast.variables.beta_k - np.transpose(CT_beta_sol)).max()
l_1 = Mast.variables.l_k 
p_1 = Mast.variables.p_k
q_1 = Mast.variables.q_k
q_pos_1 = Mast.variables.q_pos_k
alfa_1 = Mast.variables.alfa_k 
beta_1 = Mast.variables.beta_k 
imp_1 = alfa_1.sum(axis=1) 
exp_1 = beta_1.sum(axis=1)
ie_1 = imp_1-exp_1
price_comm_1 = Mast.variables.price_comm
price_IE_1 = Mast.variables.price_IE

opt1 = np.empty(TMST)
ooo = np.empty(TMST)
for i1 in range(TMST):
    opt1[i1] = (sum(y0_c[i1,:]*l_1[i1,:] + mm_c[i1,:]/2*l_1[i1,:]*l_1[i1,:]) + sum(y0_g[i1,:]*p_1[i1,:] + mm_g[i1,:]/2*p_1[i1,:]*p_1[i1,:]) #+ el_price_e[i1]*ie_1[i1]
                + 0.001*sum(abs(q_1[i1,:])) + (el_price_e[i1]+0.1)*sum(alfa_1[i1,:]) - el_price_e[i1]*sum(beta_1[i1,:]))
    ooo[i1] = (sum(y0_c[i1,:]*CT_l_sol[:,i1] + mm_c[i1,:]/2*CT_l_sol[:,i1]*CT_l_sol[:,i1]) + sum(y0_g[i1,:]*CT_p_sol[:,i1] + mm_g[i1,:]/2*CT_p_sol[:,i1]*CT_p_sol[:,i1]) + 
                (el_price_e[i1]+0.1)*CT_imp_sol[i1] - el_price_e[i1]*CT_exp_sol[i1] + 0.001*sum(abs(CT_q_sol[:,i1]))) #+ pr_i*sum(CT_alfa_sol[:,i1]) + pr_e*sum(CT_beta_sol[:,i1]))
    
gap_opt1 = opt1-CT_obj_sol
gap_opt12 = opt1-ooo

#%% Save Results
import shelve
filename='Res1_02_08.out'
var_2_save = ['price_comm_1','price_IE_1','ADMM_flag','l_1','p_1','q_1','alfa_1','beta_1']
if os.path.exists(filename+'.dat'):
    print('ERROR - file already existing!')
else:    
    my_shelf = shelve.open(filename,flag = 'n') # 'n' for new
    for key in var_2_save:
        try:
            my_shelf[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print(key)
    my_shelf.close()

#%% Community Trading
CT2_p_sol = np.zeros((g,TMST))
CT2_l_sol = np.zeros((n,TMST))
CT2_q_sol = np.zeros((n,TMST))
CT2_qcomm_sol = np.zeros((3,TMST))
CT2_alfa_sol = np.zeros((n,TMST))
CT2_beta_sol = np.zeros((n,TMST))
CT2_imp_sol = np.zeros(TMST)
CT2_exp_sol = np.zeros(TMST)
CT2_obj_sol = np.zeros(TMST)
CT2_price_sol = np.zeros((n,TMST))
CT2_price2_sol = np.zeros((2,TMST))
CT2_price3_sol = np.zeros((3,TMST))

for t in np.arange(0,TMST,24):
    temp = range(t,t+24)
    # Create a new model
    CT2_m = gb.Model("qp")
    CT2_m.setParam( 'OutputFlag', False )
    # Create variables
    p = np.array([CT2_m.addVar(lb=Pmin[i]+PV[t+k,i], ub=Pmax[i]+PV[t+k,i]) for i in range(n) for k in range(24)])
    l = np.array([CT2_m.addVar(lb=Lmin[t+k,i], ub=Lmax[t+k,i]) for i in range(n) for k in range(24)])
    q_imp = np.array([CT2_m.addVar() for k in range(24)])
    q_exp = np.array([CT2_m.addVar() for k in range(24)])
    alfa = np.array([CT2_m.addVar() for i in range(n) for k in range(24)])
    beta = np.array([CT2_m.addVar() for i in range(n) for k in range(24)])
    q_pos = np.array([CT2_m.addVar() for i in range(n) for k in range(24)])
    q = np.array([CT2_m.addVar(lb = -gb.GRB.INFINITY) for i in range(n) for k in range(24)])
    q_12 = np.array([CT2_m.addVar() for k in range(24)])
    q_13 = np.array([CT2_m.addVar() for k in range(24)])
    q_23 = np.array([CT2_m.addVar() for k in range(24)])
    q_cpp = np.array([CT2_m.addVar() for i in range(3) for k in range(24)])
    gamma_i = (el_price_e[temp] + 0.1)
    gamma_e = -el_price_e[temp]
    gamma_12 = 0.03
    gamma_23 = 0.05
    gamma_13 = 0.06
    gamma_geo = np.array([gamma_12,gamma_13,gamma_23])
    CT2_m.update()
    
    p = np.transpose(p.reshape(n,24))
    l = np.transpose(l.reshape(n,24))
    q = np.transpose(q.reshape(n,24))
    q_pos = np.transpose(q_pos.reshape(n,24))
    q_cpp = np.transpose(q_cpp.reshape(3,24))
    alfa = np.transpose(alfa.reshape(n,24))
    beta = np.transpose(beta.reshape(n,24))
    
    # Set objective: 
    obj = (sum(sum(y0_c[temp,:]*l + mm_c[temp,:]/2*l*l) + sum(y0_g[temp,:]*p + mm_g[temp,:]/2*p*p)) 
           + sum(gamma_i*q_imp + gamma_e*q_exp) + sum(q_cpp[:,0]*gamma_geo[0]+q_cpp[:,1]*gamma_geo[1]+q_cpp[:,2]*gamma_geo[2]) + sum(0.001*sum(q_pos)))
    CT2_m.setObjective(obj)
    # Add constraint
    for k in range(24):
        CT2_m.addConstr(sum(q[k,range(0,5)]) - q_12[k] - q_13[k] == 0, name="comm1[%s]"%(k))
        CT2_m.addConstr(sum(q[k,range(5,10)]) + q_12[k] - q_23[k] == 0, name="comm2[%s]"%(k))
        CT2_m.addConstr(sum(q[k,range(10,15)]) + q_13[k] + q_23[k] == 0, name="comm3[%s]"%(k))
        for i in range(n):
            CT2_m.addConstr(p[k,i] + l[k,i] + q[k,i] + alfa[k,i] - beta[k,i] == 0, name="pros[%s,%s]"% (k,i))
            CT2_m.addConstr(q[k,i] <= q_pos[k,i])
            CT2_m.addConstr(q[k,i] >= -q_pos[k,i])
        CT2_m.addConstr(sum(alfa[k,:]) - q_imp[k] == 0, name="imp_bal[%s]"%(k))
        CT2_m.addConstr(sum(beta[k,:]) - q_exp[k] == 0, name="exp_bal[%s]"%(k))
        CT2_m.addConstr(q_12[k] <= q_cpp[k,0])
        CT2_m.addConstr(q_12[k] >= -q_cpp[k,0])
        CT2_m.addConstr(q_13[k] <= q_cpp[k,1])
        CT2_m.addConstr(q_13[k] >= -q_cpp[k,1])
        CT2_m.addConstr(q_23[k] <= q_cpp[k,2])
        CT2_m.addConstr(q_23[k] >= -q_cpp[k,2])
    for i in range(n):
        CT2_m.addConstr(sum(Agg_load[temp,i] + l[:,i]) == 0)
#        for j in np.arange(0,24,window):
#            CT5_m.addConstr(sum(Agg_load[range(t+j,t+j+window),i] + l[range(j,j+window),i]) == 0)
    CT2_m.update()    
        
    CT2_m.optimize()
    for k in range(24):
        for i in range(n):
            CT2_price_sol[i,t+k] = CT2_m.getConstrByName("pros[%s,%s]"%(k,i)).Pi
            CT2_p_sol[i,t+k] = p[k,i].x
            CT2_l_sol[i,t+k] = l[k,i].x
            CT2_q_sol[i,t+k] = q[k,i].x
            CT2_alfa_sol[i,t+k] = alfa[k,i].x
            CT2_beta_sol[i,t+k] = beta[k,i].x
        CT2_price2_sol[0,t+k] = CT2_m.getConstrByName("imp_bal[%s]"%k).Pi
        CT2_price2_sol[1,t+k] = CT2_m.getConstrByName("exp_bal[%s]"%k).Pi
        CT2_price3_sol[0,t+k] = CT2_m.getConstrByName("comm1[%s]"%(k)).Pi
        CT2_price3_sol[1,t+k] = CT2_m.getConstrByName("comm2[%s]"%(k)).Pi
        CT2_price3_sol[2,t+k] = CT2_m.getConstrByName("comm3[%s]"%(k)).Pi
        CT2_imp_sol[t+k] = q_imp[k].x
        CT2_exp_sol[t+k] = q_exp[k].x
        CT2_qcomm_sol[0,t+k] = q_12[k].x
        CT2_qcomm_sol[1,t+k] = q_13[k].x
        CT2_qcomm_sol[2,t+k] = q_23[k].x

    del CT2_m

CT2_IE_sol = CT2_imp_sol - CT2_exp_sol

#%% ADMM optimisation via master-sub classes
from ADMM_master_geo3Z import ADMM_Master
o = 0.01
e = 0.00001
asynch = -100
weight = 1/n

gamma_12 = 0.03
gamma_13 = 0.06
gamma_23 = 0.05
gamma_geo = np.array([gamma_12,gamma_13,gamma_23])


Mast2 = ADMM_Master(b1, c1, Pmin, Pmax, PV, b2, c2, Load, Flex_load, tau, el_price_e, o, e, TMST, window, 0, asynch, gamma_geo)
#ADMM_summary2 = Mast2.results
#ADMM_prices2 = Mast2.prices
ADMM_flag2 = Mast2.flag

#%%
max_l_2 = abs(Mast2.variables.l_k - np.transpose(CT2_l_sol)).max()
max_p_2 = abs(Mast2.variables.p_k - np.transpose(CT2_p_sol)).max()
max_q_2 = abs(Mast2.variables.q_k - np.transpose(CT2_q_sol)).max()
max_alfa_2 = abs(Mast2.variables.alfa_k - np.transpose(CT2_alfa_sol)).max()
max_beta_2 = abs(Mast2.variables.beta_k - np.transpose(CT2_beta_sol)).max()
l_2 = Mast2.variables.l_k 
p_2 = Mast2.variables.p_k
q_2 = Mast2.variables.q_k
alfa_2 = Mast2.variables.alfa_k 
beta_2 = Mast2.variables.beta_k 
q2_com = Mast2.variables.pow_comm
price_comm_2 = Mast2.variables.price_comm
price_IE_2 = Mast2.variables.price_IE
imp_2 = alfa_2.sum(axis=1) 
exp_2 = beta_2.sum(axis=1)
ie_2 = imp_2-exp_2

opt2 = np.empty(TMST)
ooo2= np.empty(TMST)
for i1 in range(TMST):
    opt2[i1] = (sum(y0_c[i1,:]*l_2[i1,:] + mm_c[i1,:]/2*l_2[i1,:]*l_2[i1,:]) + sum(y0_g[i1,:]*p_2[i1,:] + mm_g[i1,:]/2*p_2[i1,:]*p_2[i1,:]) #+ el_price_e[i1]*ie_1[i1]
                + 0.001*sum(abs(q_2[i1,:])) + (el_price_e[i1]+0.1)*sum(alfa_2[i1,:]) - el_price_e[i1]*sum(beta_2[i1,:]) + sum(gamma_geo*abs(q2_com[i1,:])))
    ooo2[i1] = (sum(y0_c[i1,:]*CT2_l_sol[:,i1] + mm_c[i1,:]/2*CT2_l_sol[:,i1]*CT2_l_sol[:,i1]) + sum(y0_g[i1,:]*CT2_p_sol[:,i1] + mm_g[i1,:]/2*CT2_p_sol[:,i1]*CT2_p_sol[:,i1]) + 
                (el_price_e[i1]+0.1)*CT2_imp_sol[i1] - el_price_e[i1]*CT2_exp_sol[i1] + 0.001*sum(abs(CT2_q_sol[:,i1]))  + sum(gamma_geo*abs(CT2_qcomm_sol[:,i1]))) #+ pr_i*sum(CT_alfa_sol[:,i1]) + pr_e*sum(CT_beta_sol[:,i1]))
    
gap_opt2 = opt2-CT2_obj_sol
gap_opt22 = opt2-ooo2

#%% Save Results
import shelve
filename='Res2_02_08.out'
var_2_save = ['price_comm_2','price_IE_2','ADMM_flag2','l_2','p_2','q_2','q2_com','alfa_2','beta_2']
if os.path.exists(filename+'.dat'):
    print('ERROR - file already existing!')
else:    
    my_shelf = shelve.open(filename,flag = 'n') # 'n' for new
    for key in var_2_save:
        try:
            my_shelf[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print(key)
    my_shelf.close()

#%% Community Trading
CT3_p_sol = np.zeros((g,TMST))
CT3_l_sol = np.zeros((n,TMST))
CT3_q_sol = np.zeros((n,TMST))
CT3_qcomm_sol = np.zeros(TMST)
CT3_alfa_sol = np.zeros((n,TMST))
CT3_beta_sol = np.zeros((n,TMST))
CT3_imp_sol = np.zeros(TMST)
CT3_exp_sol = np.zeros(TMST)
CT3_obj_sol = np.zeros(TMST)
CT3_price_sol = np.zeros((n,TMST))
CT3_price2_sol = np.zeros((2,TMST))
CT3_price3_sol = np.zeros((3,TMST))

for t in np.arange(0,TMST,24):
    temp = range(t,t+24)
    # Create a new model
    CT3_m = gb.Model("qp")
    CT3_m.setParam( 'OutputFlag', False )
    # Create variables
    p = np.array([CT3_m.addVar(lb=Pmin[i]+PV[t+k,i], ub=Pmax[i]+PV[t+k,i]) for i in range(n) for k in range(24)])
    l = np.array([CT3_m.addVar(lb=Lmin[t+k,i], ub=Lmax[t+k,i]) for i in range(n) for k in range(24)])
    q_imp = np.array([CT3_m.addVar() for k in range(24)])
    q_exp = np.array([CT3_m.addVar() for k in range(24)])
    alfa = np.array([CT3_m.addVar() for i in range(n) for k in range(24)])
    beta = np.array([CT3_m.addVar() for i in range(n) for k in range(24)])
    q_pos = np.array([CT3_m.addVar() for i in range(n) for k in range(24)])
    q = np.array([CT3_m.addVar(lb = -gb.GRB.INFINITY) for i in range(n) for k in range(24)])
    q_12 = np.array([CT3_m.addVar() for k in range(24)])
    q_cpp = np.array([CT3_m.addVar() for k in range(24)])
    gamma_i = (el_price_e[temp] + 0.1)
    gamma_e = -el_price_e[temp]
    gamma_12 = 0.03
    CT3_m.update()
    
    p = np.transpose(p.reshape(n,24))
    l = np.transpose(l.reshape(n,24))
    q = np.transpose(q.reshape(n,24))
    q_pos = np.transpose(q_pos.reshape(n,24))
    alfa = np.transpose(alfa.reshape(n,24))
    beta = np.transpose(beta.reshape(n,24))
    
    # Set objective: 
    obj = (sum(sum(y0_c[temp,:]*l + mm_c[temp,:]/2*l*l) + sum(y0_g[temp,:]*p + mm_g[temp,:]/2*p*p)) 
           + sum(gamma_i*q_imp + gamma_e*q_exp) + sum(gamma_12*q_cpp) + sum(0.001*sum(q_pos)))
    CT3_m.setObjective(obj)
    # Add constraint
    for k in range(24):
        CT3_m.addConstr(sum(q[k,range(0,10)]) - q_12[k] == 0, name="comm1[%s]"%(k))
        CT3_m.addConstr(sum(q[k,range(10,15)]) + q_12[k] == 0, name="comm2[%s]"%(k))
        for i in range(n):
            CT3_m.addConstr(p[k,i] + l[k,i] + q[k,i] + alfa[k,i] - beta[k,i] == 0, name="pros[%s,%s]"% (k,i))
            CT3_m.addConstr(q[k,i] <= q_pos[k,i])
            CT3_m.addConstr(q[k,i] >= -q_pos[k,i])
        CT3_m.addConstr(sum(alfa[k,:]) - q_imp[k] == 0, name="imp_bal[%s]"%(k))
        CT3_m.addConstr(sum(beta[k,:]) - q_exp[k] == 0, name="exp_bal[%s]"%(k))
        CT3_m.addConstr(q_12[k] <= q_cpp[k])
        CT3_m.addConstr(q_12[k] >= -q_cpp[k])
    for i in range(n):
        CT3_m.addConstr(sum(Agg_load[temp,i] + l[:,i]) == 0)
#        for j in np.arange(0,24,window):
#            CT5_m.addConstr(sum(Agg_load[range(t+j,t+j+window),i] + l[range(j,j+window),i]) == 0)
    CT3_m.update()    
        
    CT3_m.optimize()
    for k in range(24):
        for i in range(n):
            CT3_price_sol[i,t+k] = CT3_m.getConstrByName("pros[%s,%s]"%(k,i)).Pi
            CT3_p_sol[i,t+k] = p[k,i].x
            CT3_l_sol[i,t+k] = l[k,i].x
            CT3_q_sol[i,t+k] = q[k,i].x
            CT3_alfa_sol[i,t+k] = alfa[k,i].x
            CT3_beta_sol[i,t+k] = beta[k,i].x
        CT3_price2_sol[0,t+k] = CT3_m.getConstrByName("imp_bal[%s]"%k).Pi
        CT3_price2_sol[1,t+k] = CT3_m.getConstrByName("exp_bal[%s]"%k).Pi
        CT3_price3_sol[0,t+k] = CT3_m.getConstrByName("comm1[%s]"%(k)).Pi
        CT3_price3_sol[1,t+k] = CT3_m.getConstrByName("comm2[%s]"%(k)).Pi
        CT3_imp_sol[t+k] = q_imp[k].x
        CT3_exp_sol[t+k] = q_exp[k].x
        CT3_qcomm_sol[t+k] = q_12[k].x

    del CT3_m
    
CT3_IE_sol = CT3_imp_sol - CT3_exp_sol
sum_q = CT3_q_sol.sum(axis=0)

#%% ADMM optimisation via master-sub classes
from ADMM_master_geo2Z import ADMM_Master
o = 0.01
e = 0.00001
asynch = -100
weight = 1/n

gamma_12 = 0.03
gamma_13 = 0.06
gamma_23 = 0.05
gamma_geo = gamma_12 #np.array([gamma_12,gamma_13,gamma_23])


Mast3 = ADMM_Master(b1, c1, Pmin, Pmax, PV, b2, c2, Load, Flex_load, tau, el_price_e, o, e, TMST, window, 0, asynch, gamma_geo)
#ADMM_summary3 = Mast3.results
#ADMM_prices3 = Mast3.prices
ADMM_flag3 = Mast3.flag

#%%
max_l_3 = abs(Mast3.variables.l_k - np.transpose(CT3_l_sol)).max()
max_p_3 = abs(Mast3.variables.p_k - np.transpose(CT3_p_sol)).max()
max_q_3 = abs(Mast3.variables.q_k - np.transpose(CT3_q_sol)).max()
max_alfa_3 = abs(Mast3.variables.alfa_k - np.transpose(CT3_alfa_sol)).max()
max_beta_3 = abs(Mast3.variables.beta_k - np.transpose(CT3_beta_sol)).max()
l_3 = Mast3.variables.l_k 
p_3 = Mast3.variables.p_k
q_3 = Mast3.variables.q_k
alfa_3 = Mast3.variables.alfa_k 
beta_3 = Mast3.variables.beta_k 
q3_com = Mast3.variables.pow_comm
price_comm_3 = Mast3.variables.price_comm
price_IE_3 = Mast3.variables.price_IE
imp_3 = alfa_3.sum(axis=1) 
exp_3 = beta_3.sum(axis=1)
ie_3 = imp_3-exp_3

opt3 = np.empty(TMST)
ooo3= np.empty(TMST)
for i1 in range(TMST):
    opt3[i1] = (sum(y0_c[i1,:]*l_3[i1,:] + mm_c[i1,:]/2*l_3[i1,:]*l_3[i1,:]) + sum(y0_g[i1,:]*p_3[i1,:] + mm_g[i1,:]/2*p_3[i1,:]*p_3[i1,:]) #+ el_price_e[i1]*ie_1[i1]
                + 0.001*sum(abs(q_3[i1,:])) + (el_price_e[i1]+0.1)*sum(alfa_3[i1,:]) - el_price_e[i1]*sum(beta_3[i1,:]) + gamma_geo*abs(q3_com[i1]))
    ooo3[i1] = (sum(y0_c[i1,:]*CT3_l_sol[:,i1] + mm_c[i1,:]/2*CT3_l_sol[:,i1]*CT3_l_sol[:,i1]) + sum(y0_g[i1,:]*CT3_p_sol[:,i1] + mm_g[i1,:]/2*CT3_p_sol[:,i1]*CT3_p_sol[:,i1]) + 
                (el_price_e[i1]+0.1)*CT3_imp_sol[i1] - el_price_e[i1]*CT3_exp_sol[i1] + 0.001*sum(abs(CT3_q_sol[:,i1]))  + gamma_geo*abs(CT3_qcomm_sol[i1])) #+ pr_i*sum(CT_alfa_sol[:,i1]) + pr_e*sum(CT_beta_sol[:,i1]))
    
gap_opt3 = opt3-CT3_obj_sol
gap_opt32 = opt3-ooo3


#%% Save Results
import shelve
filename='Res3_02_08.out'
var_2_save = ['price_comm_3','price_IE_3','ADMM_flag3','l_3','p_3','q_3','q3_com','alfa_3','beta_3']
if os.path.exists(filename+'.dat'):
    print('ERROR - file already existing!')
else:    
    my_shelf = shelve.open(filename,flag = 'n') # 'n' for new
    for key in var_2_save:
        try:
            my_shelf[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print(key)
    my_shelf.close()

#%% Community Trading
CT4_p_sol = np.zeros((g,TMST))
CT4_l_sol = np.zeros((n,TMST))
CT4_q_sol = np.zeros((n,TMST))
CT4_alfa_sol = np.zeros((n,TMST))
CT4_beta_sol = np.zeros((n,TMST))
CT4_imp_sol = np.zeros(TMST)
CT4_exp_sol = np.zeros(TMST)
CT4_obj_sol = np.zeros(TMST)
CT4_price_sol = np.zeros((n,TMST))
CT4_price2_sol = np.zeros((3,TMST))
CT4_price3_sol = np.zeros((n,TMST))
CT4_theta_sol = np.zeros(TMST)

for t in np.arange(0,TMST,24):
    temp = range(t,t+24)
    # Create a new model
    CT4_m = gb.Model("qp")
    CT4_m.setParam( 'OutputFlag', False )
    # Create variables
    p = np.array([CT4_m.addVar(lb=Pmin[i]+PV[t+k,i], ub=Pmax[i]+PV[t+k,i]) for i in range(n) for k in range(24)])
    l = np.array([CT4_m.addVar(lb=Lmin[t+k,i], ub=Lmax[t+k,i]) for i in range(n) for k in range(24)])
    q_imp = np.array([CT4_m.addVar() for k in range(24)])
    q_exp = np.array([CT4_m.addVar() for k in range(24)])
    alfa = np.array([CT4_m.addVar() for i in range(n) for k in range(24)])
    beta = np.array([CT4_m.addVar() for i in range(n) for k in range(24)])
    q_pos = np.array([CT4_m.addVar() for i in range(n) for k in range(24)])
    q = np.array([CT4_m.addVar(lb = -gb.GRB.INFINITY) for i in range(n) for k in range(24)])
    theta = np.array([CT4_m.addVar() for k in range(24)])
    gamma_i = (el_price_e[temp] + 0.1)
    gamma_e = -el_price_e[temp]
    gamma_m = 1.0
    CT4_m.update()
    
    p = np.transpose(p.reshape(n,24))
    l = np.transpose(l.reshape(n,24))
    q = np.transpose(q.reshape(n,24))
    q_pos = np.transpose(q_pos.reshape(n,24))
    alfa = np.transpose(alfa.reshape(n,24))
    beta = np.transpose(beta.reshape(n,24))
    
    # Set objective: 
    obj = (sum(sum(y0_c[temp,:]*l + mm_c[temp,:]/2*l*l) + sum(y0_g[temp,:]*p + mm_g[temp,:]/2*p*p)) 
           + sum(gamma_i*q_imp + gamma_e*q_exp + gamma_m*theta) + sum(0.001*sum(q_pos)))
    CT4_m.setObjective(obj)
    # Add constraint
    for k in range(24):
        CT4_m.addConstr(sum(q[k,:]) == 0, name="comm[%s]"%(k))
        for i in range(n):
            CT4_m.addConstr(p[k,i] + l[k,i] + q[k,i] + alfa[k,i] - beta[k,i] == 0, name="pros[%s,%s]"% (k,i))
            CT4_m.addConstr(q[k,i] <= q_pos[k,i])
            CT4_m.addConstr(q[k,i] >= -q_pos[k,i])
            CT4_m.addConstr(alfa[k,i] - theta[k] <= 0, name="maxx[%s,%s]"% (k,i))
        CT4_m.addConstr(sum(alfa[k,:]) - q_imp[k] == 0, name="imp_bal[%s]"%(k))
        CT4_m.addConstr(sum(beta[k,:]) - q_exp[k] == 0, name="exp_bal[%s]"%(k))
    for i in range(n):
        CT4_m.addConstr(sum(Agg_load[temp,i] + l[:,i]) == 0)
#        for j in np.arange(0,24,window):
#            CT5_m.addConstr(sum(Agg_load[range(t+j,t+j+window),i] + l[range(j,j+window),i]) == 0)
    CT4_m.update()    
        
    CT4_m.optimize()
    for k in range(24):
        for i in range(n):
            CT4_price_sol[i,t+k] = CT4_m.getConstrByName("pros[%s,%s]"%(k,i)).Pi
            CT4_price3_sol[i,t+k] = CT4_m.getConstrByName("maxx[%s,%s]"%(k,i)).Pi
            CT4_p_sol[i,t+k] = p[k,i].x
            CT4_l_sol[i,t+k] = l[k,i].x
            CT4_q_sol[i,t+k] = q[k,i].x
            CT4_alfa_sol[i,t+k] = alfa[k,i].x
            CT4_beta_sol[i,t+k] = beta[k,i].x
        CT4_price2_sol[0,t+k] = CT4_m.getConstrByName("comm[%s]"%k).Pi
        CT4_price2_sol[1,t+k] = CT4_m.getConstrByName("imp_bal[%s]"%k).Pi
        CT4_price2_sol[2,t+k] = CT4_m.getConstrByName("exp_bal[%s]"%k).Pi
        CT4_imp_sol[t+k] = q_imp[k].x
        CT4_exp_sol[t+k] = q_exp[k].x
        CT4_theta_sol[t+k] = theta[k].x

    del CT4_m

CT4_IE_sol = CT4_imp_sol - CT4_exp_sol

#%% ADMM optimisation via master-sub classes
from ADMM_master_max import ADMM_Master
o = 0.01
e = 0.00001
asynch = -100
weight = 1/n

Mast4 = ADMM_Master(b1, c1, Pmin, Pmax, PV, b2, c2, Load, Flex_load, tau, el_price_e, o, e, TMST, window, 0, asynch)
#ADMM_summary4 = Mast4.results
#ADMM_prices4 = Mast4.prices
ADMM_flag4 = Mast4.flag

#%%
max_l_4 = abs(Mast4.variables.l_k - np.transpose(CT4_l_sol)).max()
max_p_4 = abs(Mast4.variables.p_k - np.transpose(CT4_p_sol)).max()
max_q_4 = abs(Mast4.variables.q_k - np.transpose(CT4_q_sol)).max()
max_alfa_4 = abs(Mast4.variables.alfa_k - np.transpose(CT4_alfa_sol)).max()
max_beta_4 = abs(Mast4.variables.beta_k - np.transpose(CT4_beta_sol)).max()
l_4 = Mast4.variables.l_k 
p_4 = Mast4.variables.p_k
q_4 = Mast4.variables.q_k
alfa_4 = Mast4.variables.alfa_k 
beta_4 = Mast4.variables.beta_k 
theta = Mast4.variables.pow_theta
price_comm_4 = Mast4.variables.price_comm
price_IE_4 = Mast4.variables.price_IE
price_max_4 = Mast4.variables.price_max
imp_4 = alfa_4.sum(axis=1) 
exp_4 = beta_4.sum(axis=1)
ie_4 = imp_4-exp_4

opt4 = np.empty(TMST)
ooo4= np.empty(TMST)
for i1 in range(TMST):
    opt4[i1] = (sum(y0_c[i1,:]*l_4[i1,:] + mm_c[i1,:]/2*l_4[i1,:]*l_4[i1,:]) + sum(y0_g[i1,:]*p_4[i1,:] + mm_g[i1,:]/2*p_4[i1,:]*p_4[i1,:]) #+ el_price_e[i1]*ie_1[i1]
                + 0.001*sum(abs(q_4[i1,:])) + (el_price_e[i1]+0.1)*sum(alfa_4[i1,:]) - el_price_e[i1]*sum(beta_4[i1,:]) + 1.0*theta[i1])
    ooo4[i1] = (sum(y0_c[i1,:]*CT4_l_sol[:,i1] + mm_c[i1,:]/2*CT4_l_sol[:,i1]*CT4_l_sol[:,i1]) + sum(y0_g[i1,:]*CT4_p_sol[:,i1] + mm_g[i1,:]/2*CT4_p_sol[:,i1]*CT4_p_sol[:,i1]) + 
                (el_price_e[i1]+0.1)*CT4_imp_sol[i1] - el_price_e[i1]*CT4_exp_sol[i1] + 0.001*sum(abs(CT4_q_sol[:,i1]))  + 1.0*CT4_theta_sol[i1]) #+ pr_i*sum(CT_alfa_sol[:,i1]) + pr_e*sum(CT_beta_sol[:,i1]))
    
gap_opt4 = opt4-CT4_obj_sol
gap_opt42 = opt4-ooo4

#%% Save Results
import shelve
filename='Res4_02_08.out'
var_2_save = ['price_comm_4','price_IE_4','price_max_4','ADMM_flag4','l_4','p_4','q_4','alfa_4','beta_4','theta']
if os.path.exists(filename+'.dat'):
    print('ERROR - file already existing!')
else:    
    my_shelf = shelve.open(filename,flag = 'n') # 'n' for new
    for key in var_2_save:
        try:
            my_shelf[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print(key)
    my_shelf.close()
    
#%% Community Trading
CT5_p_sol = np.zeros((g,TMST))
CT5_l_sol = np.zeros((n,TMST))
CT5_q_sol = np.zeros((n,TMST))
CT5_alfa_sol = np.zeros((n,TMST))
CT5_beta_sol = np.zeros((n,TMST))
CT5_imp_sol = np.zeros(TMST)
CT5_exp_sol = np.zeros(TMST)
CT5_price_sol = np.zeros((n,TMST))
CT5_price2_sol = np.zeros((3,TMST))
CT5_price3_sol = np.zeros(TMST)
CT5_theta_sol = np.zeros(TMST)

for t in np.arange(0,TMST,24):
    temp = range(t,t+24)
    # Create a new model
    CT5_m = gb.Model("qp")
    CT5_m.setParam( 'OutputFlag', False )
    # Create variables
    p = np.array([CT5_m.addVar(lb=Pmin[i]+PV[t+k,i], ub=Pmax[i]+PV[t+k,i]) for i in range(n) for k in range(24)])
    l = np.array([CT5_m.addVar(lb=Lmin[t+k,i], ub=Lmax[t+k,i]) for i in range(n) for k in range(24)])
    q_imp = np.array([CT5_m.addVar() for k in range(24)])
    q_exp = np.array([CT5_m.addVar() for k in range(24)])
    alfa = np.array([CT5_m.addVar() for i in range(n) for k in range(24)])
    beta = np.array([CT5_m.addVar() for i in range(n) for k in range(24)])
    q_pos = np.array([CT5_m.addVar() for i in range(n) for k in range(24)])
    q = np.array([CT5_m.addVar(lb = -gb.GRB.INFINITY) for i in range(n) for k in range(24)])
    theta = CT5_m.addVar()
    gamma_i = (el_price_e[temp] + 0.1)
    gamma_e = -el_price_e[temp]
    gamma_m = 0.0
    CT5_m.update()
    
    p = np.transpose(p.reshape(n,24))
    l = np.transpose(l.reshape(n,24))
    q = np.transpose(q.reshape(n,24))
    q_pos = np.transpose(q_pos.reshape(n,24))
    alfa = np.transpose(alfa.reshape(n,24))
    beta = np.transpose(beta.reshape(n,24))
    
    # Set objective: 
    obj = (sum(sum(y0_c[temp,:]*l + mm_c[temp,:]/2*l*l) + sum(y0_g[temp,:]*p + mm_g[temp,:]/2*p*p)) 
           + sum(gamma_i*q_imp + gamma_e*q_exp) + gamma_m*theta + sum(0.001*sum(q_pos)))
    CT5_m.setObjective(obj)
    # Add constraint
    for k in range(24):
        CT5_m.addConstr(sum(q[k,:]) == 0, name="comm[%s]"%(k))
        for i in range(n):
            CT5_m.addConstr(p[k,i] + l[k,i] + q[k,i] + alfa[k,i] - beta[k,i] == 0, name="pros[%s,%s]"% (k,i))
            CT5_m.addConstr(q[k,i] <= q_pos[k,i])
            CT5_m.addConstr(q[k,i] >= -q_pos[k,i])
        CT5_m.addConstr(sum(alfa[k,:]) - q_imp[k] == 0, name="imp_bal[%s]"%(k))
        CT5_m.addConstr(sum(beta[k,:]) - q_exp[k] == 0, name="exp_bal[%s]"%(k))
        CT5_m.addConstr(q_imp[k] - theta <= 0, name="maxx[%s]"%(k))
#        CT5_m.addConstr(q_exp[k] - theta <= 0, name="maxx[%s]"%(k))
    for i in range(n):
        CT5_m.addConstr(sum(Agg_load[temp,i] + l[:,i]) == 0)
#        for j in np.arange(0,24,window):
#            CT5_m.addConstr(sum(Agg_load[range(t+j,t+j+window),i] + l[range(j,j+window),i]) == 0)
    CT5_m.update()    
        
    CT5_m.optimize()
    for k in range(24):
        for i in range(n):
            CT5_price_sol[i,t+k] = CT5_m.getConstrByName("pros[%s,%s]"%(k,i)).Pi
            CT5_p_sol[i,t+k] = p[k,i].x
            CT5_l_sol[i,t+k] = l[k,i].x
            CT5_q_sol[i,t+k] = q[k,i].x
            CT5_alfa_sol[i,t+k] = alfa[k,i].x
            CT5_beta_sol[i,t+k] = beta[k,i].x
        CT5_price2_sol[0,t+k] = CT5_m.getConstrByName("comm[%s]"%k).Pi
        CT5_price2_sol[1,t+k] = CT5_m.getConstrByName("imp_bal[%s]"%k).Pi
        CT5_price2_sol[2,t+k] = CT5_m.getConstrByName("exp_bal[%s]"%k).Pi
        CT5_price3_sol[t+k] = CT5_m.getConstrByName("maxx[%s]"%k).Pi
        CT5_imp_sol[t+k] = q_imp[k].x
        CT5_exp_sol[t+k] = q_exp[k].x
        CT5_theta_sol[t+k] = theta.x

    del CT5_m

CT5_IE_sol = CT5_imp_sol - CT5_exp_sol
sum_q = CT5_q_sol.sum(axis=0)

##%% ADMM optimisation via master-sub classes
#from ADMM_master_simple_peak import ADMM_Master
#o = 0.01
#e = 0.00001
#asynch = -100
#weight = 1/n
#
#Mast5 = ADMM_Master(b1, c1, Pmin, Pmax, PV, b2, c2, Load, Flex_load, tau, el_price_e, o, e, TMST, window, 0, asynch)
#ADMM_summary5 = Mast5.results
#ADMM_prices5 = Mast5.prices
#
##%%
#max_l_5 = abs(Mast5.variables.l_k - np.transpose(CT5_l_sol)).max()
#max_p_5 = abs(Mast5.variables.p_k - np.transpose(CT5_p_sol)).max()
#max_q_5 = abs(Mast5.variables.q_k - np.transpose(CT5_q_sol)).max()
#max_alfa_5 = abs(Mast5.variables.alfa_k - np.transpose(CT5_alfa_sol)).max()
#max_beta_5 = abs(Mast5.variables.beta_k - np.transpose(CT5_beta_sol)).max()
#
##%% Save Results
#import shelve
#filename='Res_14_07_0.out'
#var_2_save = ['Mast','Mast2','Mast3','Mast4']
#if os.path.exists(filename+'.dat'):
#    print('ERROR - file already existing!')
#else:    
#    my_shelf = shelve.open(filename,flag = 'n') # 'n' for new
#    for key in var_2_save:
#        try:
#            my_shelf[key] = globals()[key]
#        except TypeError:
#            #
#            # __builtins__, my_shelf, and imported modules can not be shelved.
#            #
#            print(key)
#    my_shelf.close()