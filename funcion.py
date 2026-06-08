import math
def area_circulo(radio):
    return math.pi * radio**2
def volumen_cilindro(radio, altura):
    return area_circulo(radio) * altura

print("Área del círculo (radio=3):", area_circulo(3))
print("Volumen del cilindro (radio=3, altura=5):", volumen_cilindro(3, 5))
