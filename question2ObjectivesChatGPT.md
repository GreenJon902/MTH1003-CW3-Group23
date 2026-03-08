Chats' summary of my objectives for question 2:

------

This script verifies that behaviour numerically by:

1. Computing a highly accurate reference solution using a very small timestep.
2. Running the midpoint solver with several larger timesteps.
3. Measuring the error in x(T) relative to the reference solution.
4. Plotting the error against h on a log-log plot.
5. Adding a reference line proportional to h^2.
6. Computing the numerical slope of the log-log graph to estimate the
   convergence rate.

------