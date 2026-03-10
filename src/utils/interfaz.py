def seleccionar_modelo() -> int:
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