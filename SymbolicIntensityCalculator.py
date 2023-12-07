import sympy as sp

# Define symbolic variables with assumptions
b, k, theta, x, y, z = sp.symbols('b k theta x y z', real=True)
I = sp.I
pi = sp.pi  # Define pi as a symbolic constant

# Define 2 * I * pi as a constant
two_i_pi = 2 * I * pi

# Define the ElectricFieldCircular expression
ElectricFieldCircular = (
    b * (
        sp.Matrix([sp.cos(theta), I, -sp.sin(theta)]) * sp.exp(I * k * (sp.sin(theta) * x + sp.cos(theta) * z)) +
        sp.Matrix([-sp.cos(theta), -I, -sp.sin(theta)]) * sp.exp(I * k * (-sp.sin(theta) * x + sp.cos(theta) * z)) +
        sp.Matrix([-I, sp.cos(theta), -sp.sin(theta)]) * sp.exp(I * k * (sp.sin(theta) * y + sp.cos(theta) * z)) +
        sp.Matrix([I, -sp.cos(theta), -sp.sin(theta)]) * sp.exp(I * k * (-sp.sin(theta) * y + sp.cos(theta) * z))
    ) + sp.Matrix([1, I, 0]) * sp.exp(I * k * z)
)

# Calculate IntensityCircular
IntensityCircular = ElectricFieldCircular.dot(ElectricFieldCircular.conjugate())

# Define symbols for Fourier transform
kx, ky, kz = sp.symbols('kx ky kz', real=True)

# Compute the inverse Fourier transform manually
FIntcf = sp.integrate(
    IntensityCircular * sp.exp(-I * (kx * x + ky * y + kz * z)),
    (x, -sp.oo, sp.oo),
    (y, -sp.oo, sp.oo),
    (z, -sp.oo, sp.oo)
)

# Print the results
print(FIntcf)
