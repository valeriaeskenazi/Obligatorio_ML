import csv

def guardar_csv(lista_resultados, output_file: str):
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label"])
        for fila in lista_resultados:
            writer.writerow(fila)
