import numpy as np
import matplotlib.pyplot as plt

# Differential Equations ---
def dxdt(x, y, mu):
    return x - x**3 / 3 - y

def dydt(x, y, mu):
    return mu**-1 * x
    

# Time-stepping Schemes ---
def forward_euler(x0: float, y0: float, h: float, count: int, mu: float) -> [(int, int)]:
    # Timesteps the Van der Pol equation `count` times with a time-step of `h` using the forward-euler scheme.
    # This uses (x0, y0) as the starting conditions.
    # Returns a numpy array of the xy-coordinates starting with (x0, y0). This will have length `count+1`.

    # Create arrays containing the coordinates at each time step, and add first coordinates
    xs = [x0]
    ys = [y0]

    for i in range(count):
        x, y = xs[-1], ys[-1]
        
        # Step x,y
        xs.append(x + h * dxdt(x, y, mu))
        ys.append(y + h * dydt(x, y, mu))

    # Convert to [(x0, y0), (x1, y1), ...] and return as a np array
    return np.array([*zip(xs, ys)])

  ######################################## Part - 1 #############################################  

x_grid = np.linspace(-3, 3, 20)
y_grid = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x_grid, y_grid)
null_range = np.linspace(-3, 3, 100)
start_x, start_y = 0.001, 0.001

plt.figure(figsize=(15, 5))

# Unstable Spiral mu=2
plt.subplot(1, 3, 1)
mu1 = 0.1
U1, V1 = dxdt(X, Y, mu1), dydt(X, Y, mu1)
plt.streamplot(X, Y, U1, V1, color='gray')

# Nullclines [x-nullcline (y = x-(x^3)/3)], [y-nullcline (x = 0)]
plt.plot(null_range, null_range - null_range**3/3, 'g--', label='x-nullcline')
plt.plot(np.zeros_like(null_range), null_range, 'm--', label='y-nullcline')

# Euler path For mu = 2
path1 = forward_euler(start_x, start_y, 0.05, 150, mu1)
plt.plot(path1[:, 0], path1[:, 1], 'r', linewidth=2)
plt.title(f"μ = {mu1} (Spiral)")
plt.grid(True, alpha=0.2)

# Unstable degenerate node mu=4
plt.subplot(1, 3, 2)
mu2 = 4
U2, V2 = dxdt(X, Y, mu2), dydt(X, Y, mu2)
plt.streamplot(X, Y, U2, V2, color='gray')

# Nullclines [x-nullcline (y = x-(x^3)/3)], [y-nullcline (x = 0)]
plt.plot(null_range, null_range - null_range**3/3, 'g--', label='x-nullcline')
plt.plot(np.zeros_like(null_range), null_range, 'm--', label='y-nullcline')

# Euler path for mu = 4
path2 = forward_euler(start_x, start_y, 0.05, 150, mu2)
plt.plot(path2[:, 0], path2[:, 1], 'r', linewidth=2)
plt.title(f"μ = {mu2} (Degenerate Node)")
plt.grid(True, alpha=0.2)

# Unstable node mu=8
plt.subplot(1, 3, 3)
mu3 = 100
U3, V3 = dxdt(X, Y, mu3), dydt(X, Y, mu3)
plt.streamplot(X, Y, U3, V3, color='gray')

# Nullclines [x-nullcline (y = x-(x^3)/3)], [y-nullcline (x = 0)]
plt.plot(null_range, null_range - null_range**3/3, 'g--', label='x-nullcline')
plt.plot(np.zeros_like(null_range), null_range, 'm--', label='y-nullcline')

# Euler path for mu = 8
path3 = forward_euler(start_x, start_y, 0.05, 150, mu3)
plt.plot(path3[:, 0], path3[:, 1], 'r', linewidth=2)
plt.title(f"μ = {mu3} (Node)")
plt.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

  ######################################## Part - 2 #############################################      

#Setting up parameters
fig, ax = plt.subplots(2,3,figsize=(18,10))

def vanderpol_plotting (mu_value):
    
    for i in range(len(mu_value)):
        mu = mu_value[i]
    
        #Phase plane
        axp = ax[0,i]
    
        #Time series
        axt = ax[1,i]
    
    #Making the function more stable for larger mu values
        h = 0.01 if mu < 4 else 0.002
        count = 5000 if mu < 4 else 20000
        t = np.linspace(0, h * count, count + 1)

    #Creating the grid for streamplot
        x_grid = np.linspace(-3, 3, 20)
        y_grid = np.linspace(-3, 3, 20)
        X, Y = np.meshgrid(x_grid, y_grid)
    
    #The maximum rage of the nullcline 
        null_range = np.linspace(-3, 3, 100)
    
    #The vector field
        U = dxdt(X,Y,mu)
        V = dydt(X,Y,mu)
    
        #Euler Trajectory and path
        #Starting at (0.0001,0.0001) for ease of view
        eu_traj = forward_euler(0.001,0.001, h, count,mu)
        x_vals = eu_traj[:, 0]
        y_vals = eu_traj[:, 1]
        axp.plot(x_vals, y_vals, 'r', linewidth=2,label = 'Euler-Trajectory')
    
        #Setting up the baseline streamplot
        axp.streamplot(X,Y,U,V, color='grey', density=1.0, linewidth=0.5)
    
        #The nullclines
        #x-nullcline (y = x-(x^3)/3)
        axp.plot(null_range, null_range - null_range**3/3, 'g--',linestyle = '--', label='x-nullcline')
        #y-nullcline (x = 0)
        axp.plot(np.zeros_like(null_range), null_range, 'black', linestyle = '--', label='y-nullcline')   
    
        #Putting the x an y axes on the plot for clearer viewing
        axp.axhline(0, color = 'darkgreen', alpha = 0.2)
        axp.axvline(0, color = 'darkgreen', alpha = 0.2)
    
        #Annotating
        axp.set_title(f"mu = {mu}")
        axp.set_xlabel("x-values")
        axp.set_ylabel("y-values")
        axp.axis('equal')
        axp.grid(True,alpha = 0.2)
        axp.legend(loc='upper right', fontsize='small')
    
        #Plotting the x and y changes according to time
        axt.plot(t, x_vals, 'r', label='x(t)')
        axt.plot(t, y_vals, 'b', alpha=0.6, label='y(t)')
        axt.set_title(f"Time Series: mu = {mu}")
        axt.set_xlabel("Time (t)")
        axt.set_ylabel("Amplitude")
        axt.grid(True, alpha=0.2)
        axt.legend(loc='upper right', fontsize='x-small')
       
    plt.tight_layout()
    plt.show

#Desired mu values in the range of 0.1 and 100
mu_value = [0.1,4,98]
vanderpol_plotting(mu_value)



