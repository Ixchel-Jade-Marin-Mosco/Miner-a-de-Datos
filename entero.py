def factorial(n):
    if n < 0:
        return "El número debe ser positivo"
    elif n == 0 or n == 1:
        return 1
    else:
        resultado = 1
        for i in range(2, n + 1):
            resultado *= i
        return resultado

print(factorial(7))  
print(factorial(6))  
