import numpy as np

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

def midpoint_method(x0: float, y0: float, h: float, count: int, mu: float) -> [(int, int)]:
    # Timesteps the Van der Pol equation `count` times with a time-step of `h` using the midpoint-method scheme.
    # This uses (x0, y0) as the starting conditions.
    # Returns a numpy array of the xy-coordinates starting with (x0, y0). This will have length `count+1`.
    
    # Create arrays containing the coordinates at each time step, and add first coordinates
    xs = [x0]
    ys = [y0]

    for i in range(count):
        x, y = xs[-1], ys[-1]
        
        # Calculate midpoint
        x_mid = xs[-1] + h / 2 * dxdt(x, y, mu)
        y_mid = ys[-1] + h / 2 * dydt(x, y, mu)
        
        # Step x,y using the midpoint
        xs.append(x + h * dxdt(x_mid, y_mid, mu))
        ys.append(y + h * dydt(x_mid, y_mid, mu))

    # Convert to [(x0, y0), (x1, y1), ...] and return as a np array
    return np.array([*zip(xs, ys)])


# Tests ---
if __name__ == "__main__": 
    print(forward_euler(1, 1, 0.01, 100, 1))
    print(midpoint_method(1, 1, 0.01, 100, 1))


    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.patches import FancyArrowPatch 

    # PGF is vector format for latex
    CREATE_PGF = True

    if CREATE_PGF:
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False
        })  
    
    fig, ax = plt.subplots()
    
    lines = [
        (forward_euler(1, 1, 0.1, 80, 1), "Forward Euler"),
        (midpoint_method(1, 1, 0.1, 80, 1), "Midpoint Method")
    ]
    
    for line_data, line_name in lines:
        
        # Plot line
        line2d = ax.plot(*zip(*line_data), label=line_name)[0]  # Only one line returned, so take the first item

        # Plot direction of line
        for i in range(0,len(line_data) - 2,10):
            pos1, pos2 = line_data[i:i+2]
            ar = FancyArrowPatch(pos1, pos2, arrowstyle="->", mutation_scale=20, color=line2d.get_color())
            ax.add_patch(ar)   

    # Annotate figure    
    #fig.suptitle("Forward Euler vs. Midpoint Method")
    #ax.set_title("80 steps with $(x_0,y_0) = (1, 1)$, $h = 0.1$, $\mu = 1$")
    ax.legend(loc='lower left')  # Show line labels

    fig.set_size_inches(7, 4)
    fig.tight_layout()
    if CREATE_PGF:
        plt.savefig("timestep-comparison-for-vanderpol.pgf")
    else:
        plt.show()

