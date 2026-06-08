# Lista de frutas
frutas = ["mango", "pera", "mango", "uva", "manzana", "fresa","uva","durazno"]
conteo = {}
for fruta in frutas:
    if fruta in conteo:
        conteo[fruta] += 1
    else:
        conteo[fruta] = 1
for fruta, cantidad in conteo.items():
    print(fruta, "=", cantidad)
