# Constants
mu = 1
h = 0.001

# Differential Equations
def dxdt(x, y):
    return x - x**3 / 3 - y

def dydt(x, y):
    return mu**-1 * x
    

def forward_euler(x0, y0):
    xs = [x0]
    ys = [y0]

    for i in range(100):
        x, y = xs[-1], ys[-1]
        xs.append(x + h * dxdt(x, y))
        ys.append(y + h * dydt(x, y))

    return xs[-1], ys[-1]

def midpoint_method(x0, y0):
    xs = [x0]
    ys = [y0]

    for i in range(100):
        x, y = xs[-1], ys[-1]
        x_mid = xs[-1] + h / 2 * dxdt(x, y)
        y_mid = ys[-1] + h / 2 * dydt(x, y)
        xs.append(x + h * dxdt(x_mid, y_mid))
        ys.append(y + h * dydt(x_mid, y_mid))

    return xs[-1], ys[-1]

print(forward_euler(1, 1))
print(midpoint_method(1, 1))

