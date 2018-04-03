import sympy
from sympy.abc import alpha, x, y, v, a, w, R, theta
from sympy import symbols, Matrix
sympy.init_printing(use_latex='mathjax', fontsize='16pt')

# Timestep
time = symbols('t')
# Distance
d = v*time + a/2*time**2
# Robot rotati1on
beta = (d/w)*sympy.tan(alpha)
# R
r = w/sympy.tan(alpha)
# State transition function
fxu = Matrix([[v+a*time], [x - r*sympy.sin(theta) + r*sympy.sin(theta+beta)],
              [y + r*sympy.cos(theta) - r*sympy.cos(theta+beta)], [theta+beta]])

print('fxu:', fxu)

F = fxu.jacobian(Matrix([v, x, y, theta]))
print('F:', F)

B, R = symbols('beta, R')
F = F.subs((d/w)*sympy.tan(alpha), B)
F = F.subs(w/sympy.tan(alpha), R)
print('F:', F)


V = fxu.jacobian(Matrix([a, alpha]))
# V = V.subs(sympy.tan(alpha)/w, 1/R)
# V = V.subs(time*v/R, B)
# V = V.subs(time*v, 'd')
print('V:', V)


tempBeta = Matrix([beta])
print(tempBeta.jacobian([alpha]))
