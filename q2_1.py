#Imports 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib

# Parameters 
mu=10 
a=0.8 
b=0.7 
Is = [0, 0.5, 1, 1.5, 2] # vary I with 0 ≤ I ≤ 2 

# Time setup 
dt=0.01 
T=100 
t=np.arange(0, T, dt) 

#Euler time step 
x=np.zeros(len(t)) 
y=np.zeros(len(t)) 
x[0]=0 
y[0]=0 


# PGF is vector image format for latex
CREATE_PGF = False

if CREATE_PGF:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False
    })  
 
for I in Is:
    for n in range(len(t)-1): 
        dx=x[n] - (1/3)*x[n]**3 - y[n] + I 
        dy=(1/mu)*(x[n] - a*y[n] + b) 
     
        x[n+1]=x[n]+dt*dx 
        y[n+1]=y[n]+dt*dy 
     
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(6, 3)

    ax1.plot(t, x, label='x(t)') 
    ax1.plot(t, y, label='y(t)') 
    ax1.legend() 
    ax1.set_xlabel('t') 
    ax1.set_title("Time-Series")
     
    #Nullclines 
    xx=np.linspace(-2, 2, 500) 
    y_null1=xx - (1/3)*xx**3 + I 
    y_null2=(xx + b)/a 
    
    #Streamplot
    xs, ys = np.meshgrid(np.linspace(min(xx), max(xx), 500), np.linspace(min([*y_null1, *y_null2]), max([*y_null1, *y_null2]), 500))
    print(xs)
    print(ys)
    ax2.streamplot(xs, ys,
                   xs - 1/3 * xs * xs * xs - ys + I,
                   1 / mu * (xs - a * ys + b),
                   color="gray",
                   linewidth=0.5)

    # Nullclines cont
    ax2.plot(xx, y_null1, '--', label='x null-cline') 
    ax2.plot(xx, y_null2, '--', label='y null-cline') 

    ax2.set_xlabel('x') 
    ax2.set_ylabel('y') 
    ax2.set_title("Phase-Plane")
    ax2.legend() 
    
    fig.suptitle(f"$I = {float(I)}$")

    fig.tight_layout()
    
    if CREATE_PGF:
        plt.savefig(f"fitz{I}.pgf")
    else:
        plt.show() 
