def select_model() -> int:
    while True:
        print("="*40)
        print(" --- MENÚ DE SELECCIÓN DEL MODELO --- ")
        print("="*40)
        print("[1] Random Forest ")
        print("[2] XGBoost")
        print("-" * 40)
        
        opcion = input("Seleccione el modelo deseado (1 o 2): ").strip()

        if opcion == "1":
            print("[INFO] Se ha seleccionado RandomForest")
            return 1
        
        elif opcion == "2":
            print("[INFO] Se ha seleccionado XGBoost")
            return 2
        
        else:
            print("[ERROR] '" + opcion + "' no es una opción válida. Por favor, intente de nuevo.")

def select_exampe() -> int:
    while True:
        print("="*60)
        print(" --- MENÚ DE SELECCIÓN DE EJEMPLO PARA INFERENCIA --- ")
        print("="*40)
        print("[1] Nuevo usuario por defeto ")
        print("[2] Subir nuevo usuario (.json)")
        print("-" * 60)
        
        opcion = input("Seleccione el modo (1 o 2): ").strip()

        if opcion == "1":
            print("[INFO] Se ha seleccionado usar el usuario por defecto")
            return 1
        
        elif opcion == "2":
            print("[INFO] Se ha seleccionado subir un nuevo usuario en formato .json")
            return 2
        
        else:
            print("[ERROR] '" + opcion + "' no es una opción válida. Por favor, intente de nuevo.")

