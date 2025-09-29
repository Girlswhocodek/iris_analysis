# -*- coding: utf-8 -*-
"""
Análisis del Dataset Iris de Fisher
Análisis exploratorio de datos y visualización
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_iris

def cargar_datos_iris():
    """
    Carga el dataset Iris y crea DataFrames para características y target
    """
    print("=" * 70)
    print("CARGANDO DATASET IRIS DE FISHER")
    print("=" * 70)
    
    # Cargar dataset desde scikit-learn
    iris = load_iris()
    
    # Crear DataFrame con las características
    X = pd.DataFrame(iris.data, columns=[
        'longitud_sepalo', 'ancho_sepalo', 
        'longitud_petalo', 'ancho_petalo'
    ])
    
    # Crear DataFrame con el target
    y = pd.DataFrame(iris.target, columns=['especies'])
    
    # Mapear números de especies a nombres
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    y['especies_nombre'] = y['especies'].map(species_map)
    
    print("✓ Datos cargados correctamente")
    print(f"✓ Forma de X (características): {X.shape}")
    print(f"✓ Forma de y (target): {y.shape}")
    
    return X, y, iris

def combinar_dataframes(X, y):
    """
    Combina características y target en un solo DataFrame
    """
    print("\n" + "=" * 70)
    print("COMBINANDO DATAFRAMES")
    print("=" * 70)
    
    # Combinar usando concat
    df = pd.concat([X, y], axis=1)
    
    print("✓ DataFrames combinados correctamente")
    print(f"✓ Forma del DataFrame combinado: {df.shape}")
    print("\nPrimeras 5 filas del DataFrame:")
    print(df.head())
    
    return df

def overview_datos(df):
    """
    Realiza un overview general de los datos
    """
    print("\n" + "=" * 70)
    print("OVERVIEW DE LOS DATOS")
    print("=" * 70)
    
    # 1. Primeras 4 muestras
    print("1. PRIMERAS 4 MUESTRAS:")
    print(df.head(4))
    
    # 2. Información general del DataFrame
    print("\n2. INFORMACIÓN DEL DATAFRAME:")
    print(df.info())
    
    # 3. Número total de muestras por especie
    print("\n3. DISTRIBUCIÓN DE ESPECIES:")
    distribucion = df['especies_nombre'].value_counts()
    print(distribucion)
    
    # 4. Verificar valores faltantes
    print("\n4. VALORES FALTANTES:")
    valores_faltantes = df.isnull().sum()
    print(valores_faltantes)
    
    # 5. Estadísticas descriptivas
    print("\n5. ESTADÍSTICAS DESCRIPTIVAS:")
    print(df.describe())

def extraccion_datos(df):
    """
    Extrae diferentes subconjuntos de datos usando loc e iloc
    """
    print("\n" + "=" * 70)
    print("EXTRACCIÓN DE DATOS")
    print("=" * 70)
    
    # 1. Extraer columna ancho_sepalo de dos maneras diferentes
    print("1. COLUMNA ANCHO_SEPALO:")
    metodo1 = df['ancho_sepalo']  # Método 1: usando nombre de columna
    metodo2 = df.iloc[:, 1]       # Método 2: usando iloc (segunda columna)
    
    print(f"✓ Método 1 (df['ancho_sepalo']): {metodo1.head(3).tolist()}")
    print(f"✓ Método 2 (df.iloc[:, 1]): {metodo2.head(3).tolist()}")
    print(f"✓ ¿Son iguales? {(metodo1.head() == metodo2.head()).all()}")
    
    # 2. Extraer filas 50 a 99 (índices 49 a 98)
    print("\n2. FILAS 50 A 99:")
    filas_50_99 = df.iloc[49:99]
    print(f"✓ Forma: {filas_50_99.shape}")
    print(f"✓ Especies en este rango: {filas_50_99['especies_nombre'].value_counts().to_dict()}")
    
    # 3. Extraer columna longitud_petalo de filas 50 a 99
    print("\n3. LONGITUD_PETALO (FILAS 50-99):")
    petalo_50_99 = df.iloc[49:99, 2]  # Tercera columna es longitud_petalo
    print(f"✓ Valores: {petalo_50_99.head(5).tolist()}")
    print(f"✓ Media: {petalo_50_99.mean():.2f}")
    
    # 4. Extraer datos donde ancho_petalo == 0.2
    print("\n4. DATOS CON ANCHO_PETALO = 0.2:")
    datos_ancho_02 = df[df['ancho_petalo'] == 0.2]
    print(f"✓ Número de muestras: {len(datos_ancho_02)}")
    print(f"✓ Especies: {datos_ancho_02['especies_nombre'].value_counts().to_dict()}")

def visualizaciones_basicas(df):
    """
    Crea visualizaciones básicas del dataset
    """
    print("\n" + "=" * 70)
    print("VISUALIZACIONES BÁSICAS")
    print("=" * 70)
    
    # Configuración para CodeSpaces
    plt.switch_backend('Agg')
    
    # Crear figura con múltiples subgráficos
    plt.figure(figsize=(20, 15))
    
    # 1. Gráfico circular de distribución de especies
    plt.subplot(2, 3, 1)
    distribucion = df['especies_nombre'].value_counts()
    colores = ['#ff9999', '#66b3ff', '#99ff99']
    plt.pie(distribucion.values, labels=distribucion.index, autopct='%1.1f%%', 
            colors=colores, startangle=90)
    plt.title('Distribución de Especies de Iris', fontweight='bold')
    
    # 2. Diagramas de caja para cada característica
    caracteristicas = ['longitud_sepalo', 'ancho_sepalo', 'longitud_petalo', 'ancho_petalo']
    
    for i, caracteristica in enumerate(caracteristicas):
        plt.subplot(2, 3, i + 2)
        datos_por_especie = [df[df['especies_nombre'] == especie][caracteristica] 
                           for especie in df['especies_nombre'].unique()]
        
        plt.boxplot(datos_por_especie, labels=df['especies_nombre'].unique())
        plt.title(f'Boxplot - {caracteristica.replace("_", " ").title()}')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('iris_visualizaciones_basicas.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Visualizaciones básicas guardadas como 'iris_visualizaciones_basicas.png'")

def visualizaciones_avanzadas(df):
    """
    Crea visualizaciones avanzadas incluyendo gráficos de violín
    """
    print("\n" + "=" * 70)
    print("VISUALIZACIONES AVANZADAS")
    print("=" * 70)
    
    plt.figure(figsize=(20, 12))
    
    caracteristicas = ['longitud_sepalo', 'ancho_sepalo', 'longitud_petalo', 'ancho_petalo']
    
    # Gráficos de violín para cada característica
    for i, caracteristica in enumerate(caracteristicas):
        plt.subplot(2, 2, i + 1)
        
        datos_por_especie = [df[df['especies_nombre'] == especie][caracteristica] 
                           for especie in sorted(df['especies_nombre'].unique())]
        
        plt.violinplot(datos_por_especie, showmeans=True, showmedians=True)
        plt.xticks([1, 2, 3], sorted(df['especies_nombre'].unique()))
        plt.title(f'Violin Plot - {caracteristica.replace("_", " ").title()}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('iris_violin_plots.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Gráficos de violín guardados como 'iris_violin_plots.png'")

def analisis_correlaciones(df):
    """
    Analiza correlaciones entre características y crea matriz de dispersión
    """
    print("\n" + "=" * 70)
    print("ANÁLISIS DE CORRELACIONES")
    print("=" * 70)
    
    # 1. Matriz de correlación
    print("1. MATRIZ DE CORRELACIÓN:")
    caracteristicas_numericas = ['longitud_sepalo', 'ancho_sepalo', 'longitud_petalo', 'ancho_petalo']
    matriz_correlacion = df[caracteristicas_numericas].corr()
    print(matriz_correlacion.round(3))
    
    # 2. Heatmap de correlaciones
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
    plt.title('Matriz de Correlación - Características del Iris', fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('iris_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Heatmap de correlaciones guardado como 'iris_correlation_heatmap.png'")
    
    # 3. Pairplot (matriz de dispersión)
    print("\n2. CREANDO MATRIZ DE DISPERSIÓN...")
    plt.figure(figsize=(12, 10))
    pairplot_fig = sns.pairplot(df, hue='especies_nombre', 
                               palette={'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'},
                               diag_kind='hist', markers=['o', 's', 'D'])
    pairplot_fig.fig.suptitle('Matriz de Dispersión - Dataset Iris', y=1.02, fontweight='bold')
    plt.savefig('iris_pairplot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Matriz de dispersión guardada como 'iris_pairplot.png'")
    
    return matriz_correlacion

def explicacion_resultados(df, matriz_correlacion):
    """
    Explica los resultados y hallazgos del análisis
    """
    print("\n" + "=" * 70)
    print("EXPLICACIÓN DE RESULTADOS Y HALLAZGOS")
    print("=" * 70)
    
    print("📊 RESUMEN DEL DATASET:")
    print(f"• Total de muestras: {len(df)}")
    print(f"• Distribución balanceada: 50 muestras por especie")
    print(f"• No hay valores faltantes")
    print(f"• 4 características numéricas continuas")
    
    print("\n🔍 HALLAZGOS PRINCIPALES:")
    
    # Análisis por especie
    print("1. ANÁLISIS POR ESPECIE:")
    for especie in df['especies_nombre'].unique():
        subset = df[df['especies_nombre'] == especie]
        print(f"   {especie.upper()}:")
        print(f"   - Longitud pétalo: {subset['longitud_petalo'].mean():.2f} ± {subset['longitud_petalo'].std():.2f}")
        print(f"   - Ancho pétalo: {subset['ancho_petalo'].mean():.2f} ± {subset['ancho_petalo'].std():.2f}")
    
    print("\n2. CORRELACIONES DESTACADAS:")
    print(f"   • Longitud pétalo - Ancho pétalo: {matriz_correlacion.loc['longitud_petalo', 'ancho_petalo']:.3f} (Muy alta)")
    print(f"   • Longitud pétalo - Longitud sépalo: {matriz_correlacion.loc['longitud_petalo', 'longitud_sepalo']:.3f} (Alta)")
    print(f"   • Ancho sépalo - Otras características: Correlaciones bajas o negativas")
    
    print("\n3. PATRONES IDENTIFICADOS:")
    print("   • Iris setosa: Pétalos notablemente más pequeños y menos variables")
    print("   • Iris versicolor: Tamaño intermedio entre setosa y virginica")
    print("   • Iris virginica: Pétalos más grandes y mayor variabilidad")
    print("   • Los pétalos son mejores discriminadores que los sépalos")
    
    print("\n4. APLICACIONES PRÁCTICAS:")
    print("   • Clasificación de especies basada en medidas morfológicas")
    print("   • Ejemplo ideal para algoritmos de clasificación supervisada")
    print("   • Benchmark para evaluar nuevos métodos de ML")

def main():
    """
    Función principal que ejecuta todo el análisis
    """
    print("🌸 ANÁLISIS EXPLORATORIO - DATASET IRIS DE FISHER")
    print("=" * 70)
    
    try:
        # 1. Cargar datos
        X, y, iris = cargar_datos_iris()
        
        # 2. Combinar DataFrames
        df = combinar_dataframes(X, y)
        
        # 3. Overview de datos
        overview_datos(df)
        
        # 4. Extracción de datos
        extraccion_datos(df)
        
        # 5. Visualizaciones básicas
        visualizaciones_basicas(df)
        
        # 6. Visualizaciones avanzadas
        visualizaciones_avanzadas(df)
        
        # 7. Análisis de correlaciones
        matriz_correlacion = analisis_correlaciones(df)
        
        # 8. Explicación de resultados
        explicacion_resultados(df, matriz_correlacion)
        
        print("\n" + "=" * 70)
        print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("=" * 70)
        print("📁 ARCHIVOS GENERADOS:")
        print("   • iris_visualizaciones_basicas.png")
        print("   • iris_violin_plots.png") 
        print("   • iris_correlation_heatmap.png")
        print("   • iris_pairplot.png")
        print("\n🎯 PRÓXIMOS PASOS SUGERIDOS:")
        print("   • Aplicar algoritmos de clasificación (KNN, SVM, Decision Trees)")
        print("   • Realizar reducción de dimensionalidad (PCA)")
        print("   • Validar modelos con cross-validation")
        
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()

# Ejecutar el análisis
if __name__ == "__main__":
    main()
