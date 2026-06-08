
numero = int(input("Escribe un número entero: "))

es_primo = True

if numero <= 1:
    es_primo = False
else:
    for i in range(2, int(numero**0.5) + 1):
        if numero % i == 0:
            es_primo = False
            break
if es_primo:
    print(f"{numero} es un número primo.")
else:
    print(f"{numero} NO es un número primo.")
