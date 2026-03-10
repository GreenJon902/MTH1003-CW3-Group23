#Imports 

import numpy as np 

import matplotlib.pyplot as plt 

 

# Parameters 

mu=10 

a=0.8 

b=0.7 

I=1 # vary I with 0 ≤ I ≤ 2 

 

# Time setup 

dt=0.01 

T=100 

t=np.arange(0, T, dt) 

 

#Euler time step 

x=np.zeros(len(t)) 

y=np.zeros(len(t)) 

x[0]=0 

y[0]=0 

 

for n in range(len(t)-1): 

    dx=x[n] - (1/3)*x[n]**3 - y[n] + I 

    dy=(1/mu)*(x[n] - a*y[n] + b) 

 

    x[n+1]=x[n]+dt*dx 

    y[n+1]=y[n]+dt*dy 

 

plt.figure() 

plt.plot(t, x, label='x(t)') 

plt.plot(t, y, label='y(t)') 

plt.legend() 

plt.xlabel('t') 

plt.savefig("Euler Time Step") 

plt.show() 

 

#Nullclines 

xx=np.linspace(-2, 2, 500) 

y_null1=xx - (1/3)*xx**3 + I 

y_null2=(xx + b)/a 

 

plt.plot(xx, y_null1, '--', label='x-nullcline') 

plt.plot(xx, y_null2, '--', label='y-nullcline') 

 

plt.xlabel('x') 

plt.ylabel('y') 

plt.legend() 

plt.savefig("Nullclines") 

plt.show() 
