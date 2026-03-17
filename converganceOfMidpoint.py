
# Convergence test for the midpoint method applied to the Van der Pol equation.

#error ≈ C * h^2, where h is the timestep size.


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from vanderpol import midpoint_method as midpointMethod

# parameters
mu = 1  # parameter mu in the differential equation
T = 10  # final integration time
x0 = 1  # initial x value
y0 = 1  # initial y value



# reference solution 
# a very small timestep is used so that this solution is extremely accurate. It is treated as the "true" solution.

hRef = 1e-5

# number of timesteps required to reach exactly t = T
countRef = int(T / hRef)

# run the midpoint solver
referenceSolution = midpointMethod(x0, y0, hRef, countRef, mu)

# extract the value of x at time T
xRef = referenceSolution[-1][0]



# define timestep sizes for the convergence experiment
hs = [1e-1, 1e-2, 1e-3, 1e-4]

errors = []


# run simulations for each timestep size
for h in hs:

    # number of steps required to reach t = T
    count = int(T / h)

    # compute numerical solution
    solution = midpointMethod(x0, y0, h, count, mu)

    # extract x(T)
    x_T = solution[-1][0]

    # compute absolute error relative to reference solution
    error = abs(x_T - xRef)

    errors.append(error)



# estimate convergence rate

# take logarithms of timestep sizes and errors
logLowercaseH = np.log(hs)
logError = np.log(errors)

# fit a straight line to log(error) vs log(h)
# slope ≈ order of the numerical method
slope, intercept = np.polyfit(logLowercaseH, logError, 1)

print("Estimated convergence rate:", slope)


CREATE_PGF = True

if CREATE_PGF:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False
    }) 

# plot results
plt.figure(figsize=(6, 4))

# plot measured errors
plt.loglog(hs, errors, 'o-', label="Measured error")

# plot reference line proportional to h^2 as per the brief
referenceLine = [h**2 for h in hs]

plt.loglog(hs, referenceLine, '--', label=r"$h^2$ reference") #chat

plt.xlabel("Timestep size (h)")
plt.ylabel(r"Error $|x(T) - x_{ref}|$")
plt.title("Convergence of the Midpoint Method")

plt.legend()

if CREATE_PGF:
    plt.savefig("convergenceOfMidpoint.pgf")
else:
    plt.show()
