def calcular_total_factura(cantidad, iva=21):
    total = cantidad * (1 + iva / 100)
    return total

print(calcular_total_factura(100))       
print(calcular_total_factura(100, 16))   
