# Assuming Qj, a, h, x, and y are defined

for i in range(len(x)):
    Qj = Qj - a * h(x[i] - y[i])
