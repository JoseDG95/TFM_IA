# ð§ª Batch de Preprocesamiento (Actualizado)
# Batch A â Mejora de contraste global y reducción de ruido
# Filtro Hampel

# Ecualización de histograma

# Transformación logarí­tmica

# Objetivo: Realzar contraste general y suavizar detalles no relevantes.

# Batch B â Mejora de contraste local con preservación de bordes
# Filtro bilateral

# CLAHE

# Corrección Gamma

# Objetivo: Resaltar tumores pequeños sin perder bordes ni detalles locales.

# Batch C â Realce de estructuras y supresión de ruido no gaussiano
# Filtro mediana

# CLAHE

# Normalización (z-score o min-max)

# Objetivo: Mejorar visibilidad de estructuras sutiles, ideal en imágenes con artefactos puntuales.

# ð¤ Modelos de Clasificación (Actualizado)
# Modelo 1 â CNN clásica
# Arquitectura convolucional construida desde cero.

# Se entrena con las imágenes preprocesadas.

# Ventaja: Captura patrones espaciales complejos propios de RMI.

# Modelo 2 â Random Forest
# Requiere una etapa de extracción de caracterí­sticas:

# Histogramas de intensidad

# Textura (Haralick, GLCM)

# Medidas estadí­sticas por zona (media, varianza, etc.)

# Ventaja: Eficaz con datasets pequeños/medianos. Interpretable.
# Nota: Ideal cuando las caracterí­sticas relevantes no son puramente espaciales.

# Modelo 3 â SVM con caracterí­sticas extraí­das
# Similar al modelo anterior, pero con SVM como clasificador.

# Ventaja: Buen desempeño con pocos datos y separación clara.
# Nota: Suele funcionar bien en imágenes mí©dicas si se extraen buenos atributos.





# from bokeh.plotting import figure, show, output_notebook
# from bokeh.models import ColumnDataSource
# from bokeh.models.tools import BoxSelectTool
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import os
# from hampel import hampel
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.ndimage import median_filter
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import shutil
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from scipy import stats
from sklearn.metrics import recall_score
from scipy.stats import f_oneway
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold

############################################################################################################
############################################################################################################
################################### DATA EXTRACTION AND DIVISION  ##########################################
############################################################################################################
############################################################################################################ 

path="C:/Users/Jose Julian/Desktop/Master_IA/Segundo Cuatrimestre/TFM/datasets"
print(os.getcwd())
os.chdir(path)
print(os.getcwd())
images_path_positive = "brain_mri_scan_images/positive"
images_path_negative = "brain_mri_scan_images/negative"
X=[]
Y=[]
for filename in os.listdir(images_path_positive):
    img_path = os.path.join(images_path_positive, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        new_size = (128, 128)  # o 224x224 si usas modelos preentrenados
        resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        X.append(resized)
        Y.append(1)
for filename in os.listdir(images_path_negative):
    img_path = os.path.join(images_path_negative, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        new_size = (128, 128)  # o 224x224 si usas modelos preentrenados
        resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        X.append(resized)
        Y.append(0)
# Mezclar los datos
X = np.array(X)
y = np.array(Y)
X, y = shuffle(X, y, random_state=42)

plt.imshow(X[3], cmap='gray')
plt.axis('off')  # Ocultar los ejes
plt.show()
print(f"Etiqueta -> {Y[3]}")


############################################################################################################
############################################################################################################
###################################   COMPUTER VISION FUNCTIONS   ##########################################
############################################################################################################
############################################################################################################ 

def hampel_filter_2d(image, window_size=5, n_sigmas=5):
    """
    Aplica un filtro de Hampel a una imagen 2D.
    - window_size: tamaño de la ventana deslizante (debe ser impar).
    - n_sigmas: níºmero de sigmas para determinar si un valor es atí­pico.
    """
    median = median_filter(image, size=window_size)
    diff = np.abs(image - median)
    mad = median_filter(diff, size=window_size)

    threshold = n_sigmas * 1.4826 * mad
    mask = diff > threshold
    filtered_image = image.copy()
    filtered_image[mask] = median[mask]
    return filtered_image.astype(np.uint8)

def umbralization(image):
    image = image.astype(np.float32)
    umbral = np.percentile(image, 93)
    ganancia = 20.0
    intensidad = 100.0
    # Aplicar función sigmoide como antes
    sigmoid = 1 / (1 + np.exp(-ganancia * (image - umbral) / 255.0))
    enhanced = image + intensidad * sigmoid
    # Reducir los píxeles fuera del percentil 90 (oscurecer más el fondo)
    reduction_mask = image < umbral
    enhanced[reduction_mask] *= 0.65  # reducir intensidad en un 40%
    return np.clip(enhanced, 0, 255).astype(np.uint8)

def central_mask(image, shrink=0.8):
    h, w = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = int(min(h, w) * shrink / 2)
    cv2.circle(mask, center, radius, 255, -1)  # Relleno completo
    return mask

def apply_central_mask(image, shrink=0.9):
    mask = central_mask(image, shrink)
    return cv2.bitwise_and(image, image, mask=mask)

def central_mask_adapted(image, margen=0.1):
    """
    Crea una máscara circular adaptada al tamaño real del cerebro dentro de la imagen.
    margen: reduce ligeramente el radio para evitar incluir el borde del cráneo.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    
    h, w = image.shape
    center_img = (w // 2, h // 2)

    # Elegir la componente que esté más cerca del centro
    best_label = None
    best_distance = float('inf')

    for i in range(1, num_labels):
        cx, cy = centroids[i]
        dist = np.sqrt((cx - center_img[0])**2 + (cy - center_img[1])**2)
        if dist < best_distance:
            best_distance = dist
            best_label = i

    # Si no se encuentra nada, devuelve máscara vacía
    if best_label is None:
        return np.ones_like(image, dtype=np.uint8) * 255

    # Obtener bounding box de la región más central (probablemente cerebro)
    x, y, w_box, h_box, _ = stats[best_label]
    r = int(min(w_box, h_box) * (1 - margen) / 2)

    mask = np.zeros_like(image, dtype=np.uint8)
    cx, cy = map(int, centroids[best_label])
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask

def mask_from_largest_component(image, margin=0.1):
    """
    Crea una máscara circular basada en la componente conectada más grande (el cerebro).
    margin: factor para reducir ligeramente el tamaño del círculo (ej. 0.1 = 10% menos)
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

    if num_labels <= 1:
        return np.ones_like(image, dtype=np.uint8) * 255  # sin regiones

    # Encuentra la región de mayor área (salta fondo: label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = 1 + np.argmax(areas)  # +1 porque stats[0] es fondo

    # Centroide y tamaño de la región cerebral
    cx, cy = map(int, centroids[max_idx])
    w_box, h_box = stats[max_idx, cv2.CC_STAT_WIDTH], stats[max_idx, cv2.CC_STAT_HEIGHT]

    # Radio reducido ligeramente por el margen
    radius = int(min(w_box, h_box) * (1 - margin) / 2)

    # Crear la máscara circular basada en ese centro y radio
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask

def apply_largest_component_mask(image, margin=0.1):
    mask = mask_from_largest_component(image, margin)
    return cv2.bitwise_and(image, image, mask=mask)

def logarithmic_transform(image, c=1):
    """
    Aplica una transformación logarítmica a una imagen.
    - image: Imagen de entrada (array de NumPy).
    - c: Constante de escalamiento (por defecto es 1).
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("La imagen debe ser un array de NumPy")
    if np.any(np.isnan(image)) or np.any(np.isinf(image)):
        raise ValueError("La imagen contiene valores NaN o inf")
    # Aplicar la transformación logarítmica
    log_image = c * np.log1p(image)
    if not isinstance(log_image, np.ndarray):
        raise ValueError("La imagen debe ser un array de NumPy")
    if np.any(np.isnan(log_image)) or np.any(np.isinf(log_image)):
        raise ValueError("La imagen contiene valores NaN o inf")
    log_image = log_image.astype(np.float32)
    normalized_image = cv2.normalize(log_image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized_image.astype(np.uint8)

def laplacian_filter(image):
    laplacian_image = cv2.Laplacian(image, cv2.CV_64F)
    # Convertir la imagen resultante a uint8 para visualización
    laplacian_image = np.uint8(np.absolute(laplacian_image))
    return laplacian_image

def multiGreyPlots (img1, img2, img3, img4, title1, title2, title3, title4):
    # plt.figure(figsize=(16, 12))
    # plt.subplot(2, 2, 1)
    # plt.imshow(img1, cmap='gray')
    # plt.title(title1)
    # plt.axis('off')
    # plt.subplot(2, 2, 2)
    # plt.imshow(img2, cmap='gray')
    # plt.title(title2)
    # plt.axis('off')
    # plt.subplot(2, 2, 3)
    # plt.imshow(img3, cmap='gray')
    # plt.title(title3)
    # plt.axis('off')
    # plt.subplot(2, 2, 4)
    # plt.imshow(img4, cmap='gray')
    # plt.title(title4, fontsize=14, fontweight='bold')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()  # Esto está bien; en modo inline no genera ventanas

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Muestra tus imágenes en cada subgráfico
    axs[0, 0].imshow(img1, cmap='gray')
    axs[0, 0].set_title(title1, fontsize=20, fontweight='bold')
    
    axs[0, 1].imshow(img2, cmap='gray')
    axs[0, 1].set_title(title2, fontsize=20, fontweight='bold')
    
    axs[1, 0].imshow(img3, cmap='gray')
    axs[1, 0].set_title(title3, fontsize=20, fontweight='bold')
    
    axs[1, 1].imshow(img4, cmap='gray')
    axs[1, 1].set_title(title4, fontsize=20, fontweight='bold')
    
    # Elimina los ejes para cada imagen (opcional)
    for ax in axs.flat:
        ax.axis('off')
    
    # Ajusta el espacio entre subgráficos
    plt.subplots_adjust(wspace=0.05, hspace=0.2)  # Ajusta estos valores según te convenga
    
    plt.show()
# Filtro de Sobel
# El filtro de Sobel es un operador de diferenciación discreta que calcula la derivada \
# de la intensidad de la imagen en las direcciones horizontal y vertical. Los bordes se \
# detectan donde hay cambios significativos en la intensidad.

def detect_edges_sobel(image):
    # Aplicar el filtro de Sobel
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # Combinar las direcciones x e y
    edges = cv2.magnitude(sobel_x, sobel_y)
    # Normalizar y convertir a uint8
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return edges

# Filtro de Canny
# El algoritmo de Canny es una técnica avanzada para la detección de bordes que\
# incluye pasos de suavizado, cálculo de gradientes, supresión de no máximos y \
# umbralización con histéresis.

def detect_edges_canny(image):
    # Aplicar el algoritmo de Canny
    edges = cv2.Canny(image, 100, 200)
    return edges


# Filtro de Laplaciano
# El filtro de Laplaciano es un operador de diferenciación de segundo orden que\
#  detecta bordes como puntos de máxima curvatura.

def detect_edges_laplacian(image):
    # Aplicar el filtro de Laplaciano
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    # Normalizar y convertir a uint8
    edges = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return edges


# Filtro de Prewitt
# El filtro de Prewitt es similar al filtro de Sobel, pero utiliza diferentes \
# coeficientes para calcular los gradientes.

def detect_edges_prewitt(image):
    # Definir los kernels de Prewitt
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    # Aplicar los kernels
    prewitt_x = cv2.filter2D(image, -1, kernel_x)
    prewitt_y = cv2.filter2D(image, -1, kernel_y)
    # Combinar las direcciones x e y
    edges = cv2.magnitude(prewitt_x, prewitt_y)
    # Normalizar y convertir a uint8
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return edges

def firstBatchImageProcessing (image, show=0):
    ##############################################################
    ######################   Filtro Hampel    ####################
    ##############################################################
    hampelImage = hampel_filter_2d(image, window_size=3, n_sigmas=2)
    ##############################################################
    ##################     Umbralization       ###################
    ##############################################################
    umbralizedImage = umbralization(hampelImage)

    ##############################################################
    #################       Mask Correction        ###############
    ##############################################################
    img_masked = apply_largest_component_mask(umbralizedImage)

    ##############################################################
    #################  Visualizar transformación   ###############
    ##############################################################
    if show == 1:
        multiGreyPlots(image, hampelImage, umbralizedImage, img_masked, \
                    "Original", "Hampel Filter", "Umbralized", "Masked")
    return img_masked

def secondBatchImageProcessing (image, show=0):
    ##############################################################
    ####################   Filtro bilateral    ###################
    ##############################################################
    bilateral_filtered = cv2.bilateralFilter(image, d=10, sigmaColor=20, sigmaSpace=10)
    ##############################################################
    ####################         CLAHE         ###################
    ##############################################################
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(4, 4))
    # clipLimit=2.0: Lí­mite de contraste; evita que el histograma se sobreamplifique y se genere ruido.
    # tileGridSize=(8, 8): Divide la imagen en 8í8 bloques y aplica ecualización localmente, luego interpola entre ellos.
    img_clahe = clahe.apply(image)
    ##############################################################
    ###################   Corrección Gamma     ###################
    ##############################################################
    rows, columns = image.shape
    img_gamma = np.zeros((rows, columns), dtype=np.uint8)
    gamma = 1.1  # Valor de gamma
    c = 255 / (255 ** gamma)  # Constante de normalización
    # Aplicar la transformación Gamma Correction píxel a píxel
    for x in range(rows):
        for y in range(columns):
            img_gamma[x, y] = np.clip(c * (img_clahe[x, y] ** gamma), 0, 255)
    ##############################################################
    #################  Visualizar transformación   ###############
    ##############################################################
    if show == 1:
        multiGreyPlots(image, bilateral_filtered, img_clahe, img_gamma, \
                    "Original", "Bilateral Filter", "CLAHE", "Gamma Correction")
    return img_gamma

def thirdBatchImageProcessing (image, show=0):
    ##############################################################
    ####################    Filtro mediana     ###################
    ##############################################################
    # Aplicar filtro mediana 3x3
    median_filtered = cv2.medianBlur(image, 3)
    ##############################################################
    ################       Umbralization       ###################
    ##############################################################

    umbralizedImage = umbralization(median_filtered)
    ##############################################################
    ###################   Corrección Gamma     ###################
    ##############################################################
    rows, columns = image.shape
    img_gamma = np.zeros((rows, columns), dtype=np.uint8)
    gamma = 1.1  # Valor de gamma
    c = 255 / (255 ** gamma)  # Constante de normalización
    # Aplicar la transformación Gamma Correction píxel a píxel
    for x in range(rows):
        for y in range(columns):
            img_gamma[x, y] = np.clip(c * (umbralizedImage[x, y] ** gamma), 0, 255)
    ##############################################################
    #################  Visualizar transformación   ###############
    ##############################################################
    if show == 1:
        multiGreyPlots(image, median_filtered, umbralizedImage, img_gamma, \
                        "Original", "Median Filter", "CLAHE", "Gamma Correction")
    return img_gamma

def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def save_image(array, path, name):
    # # Asegurarse de que el array tenga la forma correcta
    # if array.ndim == 2:
    #     array = array.reshape(array.shape[0], array.shape[1], 1)
    # elif array.ndim == 3 and array.shape[2] == 1:
    #     array = array.reshape(array.shape[0], array.shape[1])
    # elif array.ndim == 3 and array.shape[2] == 3:
    #     pass  # La imagen ya tiene 3 canales
    # else:
    #     raise ValueError("El array de la imagen no tiene la forma correcta")

    # # Asegurarse de que los valores estén en el rango correcto
    # if array.dtype != np.uint8:
    #     array = (array * 255).astype(np.uint8)

    # Convertir el array a una imagen de Pillow
    array = (array * 255).astype(np.uint8)
    img = Image.fromarray(array)

    # Guardar la imagen
    img.save(os.path.join(path, name))

# Función para crear carpetas y guardar imágenes
def save_images_to_folders(base_path, batch_name, images, y_true, y_pred, original_shape):
    folder_path = os.path.join(base_path, batch_name)
    clear_folder(folder_path)

    subfolders = [
        'no_tumor_correct',
        'no_tumor_incorrect',
        'tumor_correct',
        'tumor_incorrect'
    ]

    for subfolder in subfolders:
        clear_folder(os.path.join(folder_path, subfolder))

    for i in range(len(images)):
        if y_true[i] == 0 and y_pred[i] == 0:
            subfolder = 'no_tumor_correct'
        elif y_true[i] == 0 and y_pred[i] == 1:
            subfolder = 'no_tumor_incorrect'
        elif y_true[i] == 1 and y_pred[i] == 0:
            subfolder = 'tumor_incorrect'
        elif y_true[i] == 1 and y_pred[i] == 1:
            subfolder = 'tumor_correct'

        # Generar nombre de archivo enumerado
        filename = f'image_{i + 1}.png'

        # Reshape la imagen a su forma original antes de guardarla
        image_original_shape = images[i].reshape(original_shape)
        save_image(image_original_shape, os.path.join(folder_path, subfolder), filename)
        
############################################################################################################
############################################################################################################
################################### IMAGE TREATMENT BEFORE MODELS ##########################################
############################################################################################################
############################################################################################################     
imagesProcessedBatch1 = [] 
for index in range(0,len(X)):
    img = X[index]
    # if index in [4, 6, 13, 31, 44, 56, 70, 98, 110, 132]:
    if index in [3, 200, 500, 1000]:
        imagesProcessedBatch1.append(firstBatchImageProcessing(img, 1))
    else:
        imagesProcessedBatch1.append(firstBatchImageProcessing(img, 0))
    
imagesProcessedBatch2 = [] 
for index in range(0,len(X)):
    img = X[index]
    # if index in [4, 6, 13, 31, 44, 56, 70, 98, 110, 132]:
    if index in [3, 200, 1000, 500]:
        imagesProcessedBatch2.append(secondBatchImageProcessing(img, 1))
    else:
        imagesProcessedBatch2.append(secondBatchImageProcessing(img, 0))
    
imagesProcessedBatch3 = [] 
for index in range(0,len(X)):
    img = X[index]
    # if index in [4, 6, 13, 31, 44, 56, 70, 98, 110, 132]:
    if index in [3, 200, 1000, 500]:
        imagesProcessedBatch3.append(thirdBatchImageProcessing(img, 1))
    else:
        imagesProcessedBatch3.append(thirdBatchImageProcessing(img, 0))

multiGreyPlots(imagesProcessedBatch1[5], imagesProcessedBatch1[8], imagesProcessedBatch1[12], imagesProcessedBatch1[16], \
                "Batch 1 Image 1", "Batch 1 Image 2", "Batch 1 Image 3", "Batch 1 Image 4")
multiGreyPlots(imagesProcessedBatch2[5], imagesProcessedBatch2[8], imagesProcessedBatch2[12], imagesProcessedBatch2[16], \
                "Batch 2 Image 1", "Batch 2 Image 2", "Batch 2 Image 3", "Batch 2 Image 4")
multiGreyPlots(imagesProcessedBatch3[5], imagesProcessedBatch3[8], imagesProcessedBatch3[12], imagesProcessedBatch3[16], \
                "Batch 3 Image 1", "Batch 3 Image 2", "Batch 3 Image 3", "Batch 3 Image 4")
from PIL import Image
base_path="C:/Users/Jose Julian/Desktop/Master_IA/Segundo Cuatrimestre/TFM/Images_Processed"
image_num = 0
batch_num = 0
batch_name = ['Originals', 'Batch_1', 'Batch_2', 'Batch_3']
clear_folder(base_path)
for batch in [X, imagesProcessedBatch1, imagesProcessedBatch2, imagesProcessedBatch3]:
    for array in batch:
        img = Image.fromarray(array)
        path = base_path + "/" + batch_name[batch_num]
        name = str(image_num) + '.png'
        clear_folder(path)
        img.save(os.path.join(path, name))
        image_num = image_num + 1
    batch_num = batch_num + 1
        
    
# import time
# image = X_test[3]
# gammas = [3, 5, 7, 9]
# for gamma in gammas:
#         img_gamma = cv2.medianBlur(image, gamma)
#         plt.figure()
#         plt.imshow(img_gamma, cmap='gray')
#         plt.title(f"mediana-{gamma} ")
#         plt.axis('off')
#         plt.show()
#         time.sleep(1)
    
############################################################################################################
############################################################################################################
##################################        FUNCTIONS DECLARATION      #######################################
############################################################################################################
############################################################################################################ 

# Generar la matriz de confusión
def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    # plt.figure(figsize=(10, 8))
    # plt.imshow(cm_df, cmap='Blues', interpolation='nearest')
    # plt.colorbar()
    # plt.title(title)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    # plt.yticks(np.arange(len(class_names)), class_names)

    # for i in range(len(class_names)):
    #     for j in range(len(class_names)):
    #         plt.text(j, i, cm_df.iloc[i, j], ha='center', va='center', color='black')

    # plt.tight_layout()
    # plt.show()
    return cm
    
    
def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


# def save_image(array, path, name):
#     plt.imsave(os.path.join(path, name), array, cmap='gray')    

def save_image(array, path, name):

    array = (array * 255).astype(np.uint8)
    img = Image.fromarray(array)

    # Guardar la imagen
    img.save(os.path.join(path, name))

# Función para crear carpetas y guardar imágenes
def save_images_to_folders(base_path, batch_name, images, y_true, y_pred, original_shape):
    folder_path = os.path.join(base_path, batch_name)
    clear_folder(folder_path)

    subfolders = [
        'no_tumor_correct',
        'no_tumor_incorrect',
        'tumor_correct',
        'tumor_incorrect'
    ]

    for subfolder in subfolders:
        clear_folder(os.path.join(folder_path, subfolder))

    for i in range(len(images)):
        if y_true[i] == 0 and y_pred[i] == 0:
            subfolder = 'no_tumor_correct'
        elif y_true[i] == 0 and y_pred[i] == 1:
            subfolder = 'no_tumor_incorrect'
        elif y_true[i] == 1 and y_pred[i] == 0:
            subfolder = 'tumor_incorrect'
        elif y_true[i] == 1 and y_pred[i] == 1:
            subfolder = 'tumor_correct'

        # Generar nombre de archivo enumerado
        filename = f'image_{i + 1}.png'

        # Reshape la imagen a su forma original antes de guardarla
        image_original_shape = images[i].reshape(original_shape)
        save_image(image_original_shape, os.path.join(folder_path, subfolder), filename)    
        
def recall_metric(y_true, y_pred):
    # y_true: shape (batch,), valores 0/1
    # y_pred: shape (batch, 2), probabilidades softmax
    y_true = tf.cast(tf.squeeze(y_true), tf.int64)
    y_pred_labels = tf.argmax(y_pred, axis=1, output_type=tf.int64)

    tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1),
                                              tf.equal(y_pred_labels, 1)), tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1),
                                              tf.not_equal(y_pred_labels, 1)), tf.float32))
    return tf.math.divide_no_nan(tp, tp + fn)

# Nombre corto para que history guarde 'recall' y 'val_recall'
recall_metric.__name__ = 'recall'

############################################################################################################
############################################################################################################
################################        CNN MODEL DECLARATION      #########################################
############################################################################################################
############################################################################################################ 

# results_summary = []
# titles =  ["Batch_1", "Batch_2", "Batch_3"]
# it=0   
# X_train=0
# X_test=0
# y_train=0
# y_test=0
# base_path="C:/Users/Jose Julian/Desktop/Master_IA/Segundo Cuatrimestre/TFM/Resultados_clasificacion_CNN"
results_summary = []
titles =  ["Batch_1", "Batch_2", "Batch_3"]
it=0 
base_path="C:/Users/Jose Julian/Desktop/Master_IA/Segundo Cuatrimestre/TFM/Resultados_clasificacion_CNN"

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for x in [imagesProcessedBatch1, imagesProcessedBatch2, imagesProcessedBatch3]:
    original_shape = x[0].shape
    x = np.expand_dims(x, axis=-1).astype('float32') / 255.

    fold_metrics = []
    cms = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(x, y), start=1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        X_train, X_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Modelo
        model = Sequential([
            Input(shape=(128, 128, 1)),
            Conv2D(64, (3, 3), padding='valid', activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), padding='valid', activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), padding='valid', activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(2, activation='softmax')
        ])

        # *** Añade recall como métrica principal de seguimiento
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy', recall_metric])

        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=64,
            validation_split=0.2,
            verbose=1
        )

        # *** UNA sola gráfica con Accuracy y Recall (train/val)
        hist = history.history
        plt.figure()
        plt.plot(hist['accuracy'], label='Train Acc')
        plt.plot(hist['val_accuracy'], label='Val Acc')
        plt.plot(hist['recall'], label='Train Recall')
        plt.plot(hist['val_recall'], label='Val Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title(f'{titles[it]} - Fold {fold} | Accuracy & Recall')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # Evaluación en test del fold
        test_loss, test_acc, test_recall = model.evaluate(X_test, y_test, verbose=0)
        print(f'Test accuracy (fold {fold}): {test_acc:.4f}')

        preds = model.predict(X_test, verbose=0)
        y_pred = np.argmax(preds, axis=1)

        report = classification_report(y_test, y_pred, target_names=['Tumor', 'Sano'], output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=[0,1])

        fold_metrics.append({
            'Fold': fold,
            'Accuracy': test_acc,
            'Precision_Tumor': report['Tumor']['precision'],
            'Recall_Tumor': report['Tumor']['recall'],
            'F1_Tumor': report['Tumor']['f1-score']
        })
        cms.append(cm)

    # Resumen por batch
    df_fold = pd.DataFrame(fold_metrics)
    avg = df_fold[['Accuracy', 'Precision_Tumor', 'Recall_Tumor', 'F1_Tumor']].mean().to_dict()
    print(f"\nMedia {titles[it]} en {n_splits} folds:\n{avg}")

    cm_sum = np.sum(cms, axis=0)
    results_summary.append({
        'Batch': titles[it],
        'Accuracy': avg['Accuracy'],
        'Precision': avg['Precision_Tumor'],
        'Recall': avg['Recall_Tumor'],
        'F1-score': avg['F1_Tumor'],
        'Confusion_Matrix': cm_sum.tolist()
    })

    it += 1

df_results = pd.DataFrame(results_summary)
print(df_results)
df_results.to_csv("cnn_results_KFold.csv", index=False)

############################################################################################################
############################################################################################################
##############################        SVM + PCA MODEL DECLARATION      #####################################
############################################################################################################
############################################################################################################ 

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from PIL import Image
base_path="C:/Users/Jose Julian/Desktop/Master_IA/Segundo Cuatrimestre/TFM/Resultados_clasificacion_SVM"

# Funciones existentes
def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)



# Resto del código sin cambios
it = 1
for x in [imagesProcessedBatch1, imagesProcessedBatch2, imagesProcessedBatch3]:
    # Dividir los datos en conjuntos de entrenamiento y prueba
    x = np.array(x)  # o cualquier batch
    x = x.astype('float32') / 255.0
    original_shape = x.shape[1:]  # Guardar la forma original de las imágenes
    x = x.reshape(len(x), -1)  # Aplana cada imagen
    pca = PCA()
    pca.fit(x)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()
    # Graficar la varianza explicada acumulada
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.xlabel('Número de Componentes Principales')
    plt.ylabel('Varianza Explicada Acumulada')
    plt.title(f'Varianza Explicada Acumulada vs. Número de Componentes Principales para Batch {it}')
    plt.grid()
    plt.show()
    it += 1

print("Mediante el método de la varianza explicada, en el gráfico observamos que el número óptimo de componentes para PCA es 750")

# results_table = []
# results_table_test = []
# titles = ["Batch_1", "Batch_2", "Batch_3"]
# it = 1
# clear_folder(base_path)

# for x in [imagesProcessedBatch1, imagesProcessedBatch2, imagesProcessedBatch3]:
#     # Dividir los datos en conjuntos de entrenamiento y prueba
#     x = np.array(x)  # o cualquier batch
#     X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#     X_train = X_train.astype('float32') / 255.0
#     X_test = X_test.astype('float32') / 255.0
#     original_shape = X_train.shape[1:]  # Guardar la forma original de las imágenes
#     X_train = X_train.reshape(len(X_train), -1)  # Aplana cada imagen
#     X_test = X_test.reshape(len(X_test), -1)  # Aplana cada imagen

#     ############################### PCA #############################################################

#     pca = PCA(n_components=750)  # Reducir componentes principales
#     X_train_pca = pca.fit_transform(X_train)  # aprende y transforma
#     X_test_pca = pca.transform(X_test)  # solo transforma usando lo aprendido
#     pca_df = pd.DataFrame(data=X_test_pca)

#     ###########
#     # Visualizar los resultados de PCA
#     ###########

#     # print(pca_df)
#     # Mostrar la varianza explicada por cada componente principal
#     explained_variance = pca.explained_variance_ratio_
#     # print("Varianza explicada por cada componente principal:", explained_variance)
#     # Mostrar los componentes principales
#     # print("Componentes principales:")
#     # print(pca.components_)

#     ############################### SVM #############################################################

#     ## Entrena un conjunto de modelos de SVM que utilicen todas las variables del dataset, con C = 0.1 hasta 1000 con cinco \
#     ##       valores diferentes; kernel radial, y gamma con valor scale.
#     # Definir los valores de C y gamma
#     # C_values = 10**np.linspace(-1, 2, 3)  # [0.1, 1, 10, 100, 1000]
#     C_values = [0.1, 5, 10]
#     gamma_value = 'scale'  # 'scale' es una cadena y está bien así
#     kernels = ['poly', 'rbf', 'sigmoid']

#     modelos = []
#     print("####################################")
#     print(f"Batch_{it}")
#     print("####################################")
#     # Crear modelos para cada kernel y cada valor de C
#     for kernel in kernels:
#         for C in C_values:
#             if kernel == 'poly':
#                 modelos.append(SVC(kernel='poly', degree=3, C=C, gamma=gamma_value))
#                 modelos.append(SVC(kernel='poly', degree=4, C=C, gamma=gamma_value))
#             else:
#                 modelos.append(SVC(kernel=kernel, C=C, gamma=gamma_value))

#     # Entrenar los modelos
#     modelos_trained = []
#     for modelo in modelos:
#         print(f"entrenando modelo {modelo.kernel} C-{modelo.C}")
#         modelo.fit(X_train_pca, y_train)        
#         modelos_trained.append(modelo)

#     for modelo in modelos_trained:
#         print(f"testeando modelo {modelo.kernel} C-{modelo.C}")
#         y_pred_train = modelo.predict(X_train_pca)
#         cm2 = confusion_matrix(y_train, y_pred_train, labels=modelo.classes_)
#         disp = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=modelo.classes_)
#         disp.plot(cmap='Blues')
#         plt.xticks(rotation=90)
#         if modelo.kernel == 'poly':
#             plt.title(f"Modelo SVM Kernel-{modelo.kernel} Degree-{modelo.degree} C-{modelo.C} Batch_{it}")
#         else:
#             plt.title(f"Modelo SVM Kernel-{modelo.kernel} C-{modelo.C} Batch_{it}")
#         plt.show()
#         accuracy_train = accuracy_score(y_train, y_pred_train)
#         f1_macro_train = f1_score(y_train, y_pred_train, average='macro')
#         f1_micro_train = f1_score(y_train, y_pred_train, average='micro')
#         recall_train = recall_score(y_train, y_pred_train, average='macro')
#         # Número total de clasificaciones correctas (suma de la diagonal)
#         correctas = np.trace(cm2)  # Suma de los valores en la diagonal
#         # Número total de clasificaciones incorrectas (suma de todo menos la diagonal)
#         incorrectas = np.sum(cm2) - correctas

#         print(f"Clasificaciones correctas de Modelo SVM Kernel-{modelo.kernel} Degree-{modelo.degree} C-{modelo.C} -> {correctas}")
#         print(f"Clasificaciones incorrectas de Modelo SVM Kernel-{modelo.kernel} Degree-{modelo.degree} C-{modelo.C} -> {incorrectas}")
#         if modelo.kernel == 'poly':
#             print(f"Acquracy for Modelo SVM Kernel-{modelo.kernel} Degree-{modelo.degree} C-{modelo.C} -> {accuracy_train}")
#             print(f"F1 Score 'macro' for Modelo SVM Kernel-{modelo.kernel} Degree-{modelo.degree} C-{modelo.C} -> {f1_macro_train}")
#             print(f"F1 Score 'micro' for Modelo SVM Kernel-{modelo.kernel} Degree-{modelo.degree} C-{modelo.C} -> {f1_micro_train}")
#             print(f"Recall 'macro' for Modelo SVM Kernel-{modelo.kernel} Degree-{modelo.degree} C-{modelo.C} -> {recall_train}")
#         else:
#             print(f"Acquracy for Modelo SVM Kernel-{modelo.kernel} C-{modelo.C} -> {accuracy_train}")
#             print(f"F1 Score 'macro' for Modelo SVM Kernel-{modelo.kernel} C-{modelo.C} -> {f1_macro_train}")
#             print(f"F1 Score 'micro' for Modelo SVM Kernel-{modelo.kernel} C-{modelo.C} -> {f1_micro_train}")
#             print(f"Recall 'macro' for Modelo SVM Kernel-{modelo.kernel} C-{modelo.C} -> {recall_train}")
#         result_row = {
#             'Batch': f'Batch_{it}',
#             'Kernel': modelo.kernel,
#             'C': modelo.C,
#             'Accuracy': accuracy_train,
#             'F1_macro': f1_macro_train,
#             'F1_micro': f1_micro_train,
#             'Recall_macro': recall_train,
#             'Correctas': correctas,
#             'Incorrectas': incorrectas
#         }

#         # Añadir el grado solo si kernel es polynomial
#         if modelo.kernel == 'poly':
#             result_row['Degree'] = modelo.degree
#         else:
#             result_row['Degree'] = None

#         results_table.append(result_row)

#     results_df = pd.DataFrame(results_table)
#     results_df = results_df.sort_values(by='Accuracy', ascending=False)  # ordenar si quieres
#     print(results_df)
#     results_df.to_csv(f"svm_results_batch{it}.csv", index=False)
#     best_models = []
#     for modelo_search in modelos_trained:
#         if (modelo_search.kernel == 'rbf' and modelo_search.C == 100) or (modelo_search.kernel == 'poly' and modelo_search.degree == 4 and modelo_search.C == 100) or (modelo_search.kernel == 'rbf' and modelo_search.C == 3.1622776601683795) or (modelo_search.kernel == 'poly' and modelo_search.degree == 4 and modelo_search.C == 3.1622776601683795):
#             best_models.append(modelo_search)
#     for modelo in best_models:
#         y_pred_test = modelo.predict(X_test_pca)
#         cm2 = confusion_matrix(y_test, y_pred_test, labels=modelo.classes_)
#         disp = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=modelo.classes_)
#         disp.plot(cmap='Blues')
#         plt.xticks(rotation=90)
#         plt.title(f"Modelo SVM Kernel-{modelo.kernel} Degree-{modelo.degree} C-{modelo.C}")
#         accuracy_test = accuracy_score(y_test, y_pred_test)
#         f1_macro_test = f1_score(y_test, y_pred_test, average='macro')
#         f1_micro_test = f1_score(y_test, y_pred_test, average='micro')
#         recall_test = recall_score(y_test, y_pred_test, average='macro')
#         # Número total de clasificaciones correctas (suma de la diagonal)
#         correctas = np.trace(cm2)  # Suma de los valores en la diagonal
#         # Número total de clasificaciones incorrectas (suma de todo menos la diagonal)
#         incorrectas = np.sum(cm2) - correctas
#         print(f"Clasificaciones correctas de Modelo SVM Kernel-{modelo.kernel} Degree-{modelo.degree} C-{modelo.C} -> {correctas}")
#         print(f"Clasificaciones incorrectas de Modelo SVM Kernel-{modelo.kernel} Degree-{modelo.degree} C-{modelo.C} -> {incorrectas}")
#         print(f"Acquracy for Modelo SVM Kernel-{modelo.kernel} Degree-{modelo.degree} C-{modelo.C} -> {accuracy_test}")
#         print(f"F1 Score 'macro' for Modelo SVM Kernel-{modelo.kernel} Degree-{modelo.degree} C-{modelo.C} -> {f1_macro_test}")
#         print(f"F1 Score 'micro' for Modelo SVM Kernel-{modelo.kernel} Degree-{modelo.degree} C-{modelo.C} -> {f1_micro_test}")
#         print(f"Recall 'macro' for Modelo SVM Kernel-{modelo.kernel} Degree-{modelo.degree} C-{modelo.C} -> {recall_test}")
#         print("Train Report:\n", classification_report(y_test, y_pred_test))
#         result_row = {
#             'Batch': f'Batch_{it}',
#             'Kernel': modelo.kernel,
#             'C': modelo.C,
#             'Accuracy': accuracy_test,
#             'F1_macro': f1_macro_test,
#             'F1_micro': f1_micro_test,
#             'Recall_macro': recall_test,
#             'Correctas': correctas,
#             'Incorrectas': incorrectas,
#             "Confusion_Matrix": cm2.tolist()
#         }
#         results_table_test.append(result_row)
        
#         # Guardar imágenes en carpetas correspondientes
#         save_images_to_folders(base_path, f'Batch_{it}', X_test, y_test, y_pred_test, original_shape)

#     it = it + 1

# results_df = pd.DataFrame(results_table_test)
# results_df = results_df.sort_values(by='Accuracy', ascending=False)  # ordenar si quieres
# print(results_df)
# results_df.to_csv(f"svm_results_TEST.csv", index=False)





from sklearn.metrics import make_scorer, recall_score
import seaborn as sns
import os, shutil
# Crear carpeta si no existe
def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

results_table_cv = []     # Para guardar resultados de GridSearchCV
results_table_test = []   # Para guardar resultados de evaluación en test
titles = ["Batch_1", "Batch_2", "Batch_3"]
it = 1
clear_folder(base_path)

param_grid = [
    {'kernel': ['rbf', 'sigmoid'], 'C': [0.1, 5, 10], 'gamma': ['scale']},
    {'kernel': ['poly'],           'C': [0.1, 5, 10], 'degree': [3, 4], 'gamma': ['scale']},
]

recall_macro = make_scorer(recall_score, average='macro')

for x, title in zip([imagesProcessedBatch1, imagesProcessedBatch2, imagesProcessedBatch3], titles):
    print(f"\n=== Procesando {title} ===")

    # 1. División de datos
    x = np.array(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    original_shape = X_train.shape[1:]
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    # 2. PCA
    pca = PCA(n_components=750)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # 3. GridSearchCV
    svc = SVC(gamma='scale')
    grid_search = GridSearchCV(
        estimator=svc,
        param_grid=param_grid,
        scoring=recall_macro,
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train_pca, y_train)

    # # 4. Guardar resultados del CV en tabla + CSV
    # cv_results_df = pd.DataFrame(grid_search.cv_results_)
    # cv_results_df['Batch'] = title
    # # Crear una etiqueta que combina kernel y degree (si aplica)
    # cv_results_df['kernel_label'] = cv_results_df.apply(
    #     lambda row: f"{row['param_kernel']}_deg{row['param_degree']}"
    #     if row['param_kernel'] == 'poly' else row['param_kernel'],
    #     axis=1
    # )
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df['Batch'] = title
    
    # Degree limpio (solo para poly). OJO: convertir a Int64 (permite NA)
    cv_results_df['degree_clean'] = np.where(
        cv_results_df['param_kernel'] == 'poly',
        cv_results_df['param_degree'].astype('Int64'),
        pd.NA
    )
    cv_sorted = cv_results_df.sort_values('mean_test_score', ascending=False)
    cv_unique = cv_sorted.drop_duplicates(
        subset=['param_kernel', 'param_C', 'degree_clean'],
        keep='first'
    )
    

    results_table_cv.append(cv_unique)


    # Graficar los resultados
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=cv_unique, x='param_C', y='mean_test_score', hue='param_kernel', marker='o')
    plt.title(f'{title} - CV Recall Macro')
    plt.xlabel('C')
    plt.ylabel('Mean CV Score (Recall_macro)')
    plt.grid(True)
    plt.legend(title='Kernel')
    plt.show()

    # 5. Seleccionar los 2 mejores modelos por `mean_test_score`
    top_models = cv_unique.sort_values(by='mean_test_score', ascending=False).head(2)

    for _, row in top_models.iterrows():
        kernel = row['param_kernel']
        C = float(row['param_C'])
    
        if kernel == 'poly':
            degree = int(row['degree_clean'])     # <- aquí el cast a int
            model = SVC(kernel='poly', C=C, degree=degree, gamma='scale')
        else:
            model = SVC(kernel=kernel, C=C, gamma='scale')  # no pases degree
    
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)
    
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, average='macro')
        cm  = confusion_matrix(y_test, y_pred)
    
        results_table_test.append({
            'Batch': title,
            'Kernel': kernel,
            'C': C,
            'Degree': degree if kernel=='poly' else None,
            'Accuracy': acc,
            'Recall_macro': rec,
            'Confusion_Matrix': cm.tolist()
        })

    it += 1

# 6. Guardar todos los resultados de GridSearchCV en CSV
all_cv_results_df = pd.concat(results_table_cv, ignore_index=True)
all_cv_results_df.to_csv(os.path.join(base_path, "svm_cv_results.csv"), index=False)

# 7. Guardar resultados de evaluación final (test) en CSV
final_test_df = pd.DataFrame(results_table_test)
final_test_df.to_csv(os.path.join(base_path, "svm_test_results.csv"), index=False)

print("\n✅ Todo listo. Resultados guardados:")
print("- Resultados de CV → svm_cv_results.csv")
print("- Evaluación en test final → svm_test_results.csv")
############################################################################################################
############################################################################################################
#######################################           XGBoost        ###########################################
############################################################################################################
############################################################################################################ 

from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
import os
import shutil
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

# def clear_folder(folder_path):
#     if os.path.exists(folder_path):
#         shutil.rmtree(folder_path)
#     os.makedirs(folder_path)

# # def save_image(image_array, path, name):
# #     img = Image.fromarray(image_array)
# #     image_original_shape = img.reshape(original_shape)
# #     img.save(os.path.join(path, name))


# def save_image(array, path, name):


#     # Convertir el array a una imagen de Pillow
#     array = (array * 255).astype(np.uint8)
#     img = Image.fromarray(array)

#     # Guardar la imagen
#     img.save(os.path.join(path, name))


# base_path = "C:/Users/Jose Julian/Desktop/Master_IA/Segundo Cuatrimestre/TFM/Resultados_clasificacion_XGBoost"
# clear_folder(base_path)

# results = []
# titles = ["Batch_1", "Batch_2", "Batch_3"]
# batches = [imagesProcessedBatch1, imagesProcessedBatch2, imagesProcessedBatch3]

# for it, x_ in enumerate(batches, start=1):
#     # X = np.array(x_)
#     # X = X.reshape(X.shape[0], -1)
#     # original_images = X  # para guardar las que fallan
#     # y_batch = y  # cambia esto si usas y1, y2, y3 por batch

#     # X_train, X_test, y_train, y_test = train_test_split(X, y_batch, test_size=0.2, random_state=42)

#     # scaler = StandardScaler()
#     # X_train = scaler.fit_transform(X_train)
#     # X_test = scaler.transform(X_test)
    
#     x = np.array(x_)  # o cualquier batch
#     X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#     X_train = X_train.astype('float32') / 255.0
#     X_test = X_test.astype('float32') / 255.0
#     original_shape = X_train.shape[1:]  # Guardar la forma original de las imágenes
#     X_train = X_train.reshape(len(X_train), -1)  # Aplana cada imagen
#     X_test = X_test.reshape(len(X_test), -1)  # Aplana cada imagen

#     ############################### PCA #############################################################

#     pca = PCA(n_components=750)  # Reducir componentes principales
#     X_train_pca = pca.fit_transform(X_train)  # aprende y transforma
#     X_test_pca = pca.transform(X_test)  # solo transforma usando lo aprendido
#     pca_df = pd.DataFrame(data=X_test_pca)

#     param_grid = {
#         'max_depth': [4, 7],
#         'learning_rate': [0.01, 0.07],
#         'n_estimators': [50, 100],
#         'subsample': [0.9],
#         'colsample_bytree': [0.9]
#     }

#     model = xgb.XGBClassifier(objective='binary:logistic', seed=42)

#     grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
#     encoder = OneHotEncoder(sparse_output=False)
#     y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
#     grid_search.fit(X_train_pca, y_train_onehot)

#     best_model = grid_search.best_estimator_
#     best_params = grid_search.best_params_

#     y_pred = best_model.predict(X_test_pca)
#     encoder = OneHotEncoder(sparse_output=False)
#     y_test_onehot = encoder.fit_transform(y_test.reshape(-1, 1))
#     acc = accuracy_score(y_test_onehot, y_pred)
#     f1_macro = f1_score(y_test_onehot, y_pred, average='macro')
#     f1_micro = f1_score(y_test_onehot, y_pred, average='micro')
#     recall_macro = recall_score(y_test_onehot, y_pred, average='macro')
#     y_pred_labels = np.argmax(y_pred, axis=1)
#     cm = confusion_matrix(y_test, y_pred_labels)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=modelo.classes_)
#     disp.plot(cmap='Blues')

#     results.append({
#         "Batch": f"Batch_{it}",
#         "Accuracy": acc,
#         "F1_macro": f1_macro,
#         "F1_micro": f1_micro,
#         "Recall_macro": recall_macro,
#         "Best_Params": best_params,
#         "Confusion_Matrix": cm.tolist()
#     })

#     save_images_to_folders(base_path, f'Batch_{it}', X_test, y_test, y_pred_labels, original_shape)
#     # # Guardar imágenes mal clasificadas
#     # for i, (true_label, pred_label) in enumerate(zip(y_test, y_pred_labels)):
#     #     if true_label != pred_label:
#     #         error_type = "Positivos" if true_label == 1 else "Negativos"
#     #         folder_path = os.path.join(base_path, f"Batch_{it}", "Fallos", error_type)
#     #         os.makedirs(folder_path, exist_ok=True)
#     #         save_image(original_images[i], folder_path, f"img_{i}_T{true_label}_P{pred_label}.png")

#     results_df = pd.DataFrame(results)
#     results_df = results_df.sort_values(by='Accuracy', ascending=False)  # ordenar si quieres
#     print(results_df)
#     results_df.to_csv(f"xgboost_results_batch{it}.csv", index=False)



############################################################################################################
#########################################  XGBoost + CV (Recall)  ##########################################
############################################################################################################
import os, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    make_scorer, recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

# ---------------------- utils ----------------------
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def save_image(array, path, name):
    array = (array * 255).astype(np.uint8)
    img = Image.fromarray(array)
    img.save(os.path.join(path, name))

# ---------------------- setup ----------------------
base_path = r"C:/Users/Jose Julian/Desktop/Master_IA/Segundo Cuatrimestre/TFM/Resultados_clasificacion_XGBoost"
clear_folder(base_path)

titles  = ["Batch_1", "Batch_2", "Batch_3"]
batches = [imagesProcessedBatch1, imagesProcessedBatch2, imagesProcessedBatch3]

# Métrica principal
recall_macro_scorer = make_scorer(recall_score, average="macro")

# Grid MODERADO (pensando en tu PC)
param_grid = {
    "max_depth": [4, 6],
    "learning_rate": [0.1, 0.05],
    "n_estimators": [200, 400, 600],
    "subsample": [0.9],
    "colsample_bytree": [0.9],
    "min_child_weight": [1, 3],
    "gamma": [0]  # fijo para no crecer el grid
}


# Tablas a guardar
all_cv_results   = []
all_test_results = []

for title, Ximgs in zip(titles, batches):
    print(f"\n=== Procesando {title} ===")

    # ---------------------- split + flatten + PCA ----------------------
    Ximgs = np.array(Ximgs)
    X_tr, X_te, y_tr, y_te = train_test_split(Ximgs, y, test_size=0.2, random_state=42, stratify=y)
    X_tr = X_tr.astype("float32")/255.0
    X_te = X_te.astype("float32")/255.0
    original_shape = X_tr.shape[1:]
    X_tr = X_tr.reshape(len(X_tr), -1)
    X_te = X_te.reshape(len(X_te), -1)

    pca = PCA(n_components=750)
    X_tr_pca = pca.fit_transform(X_tr)
    X_te_pca = pca.transform(X_te)


    xgb_base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )

    # ---------------------- GridSearchCV (CV=3, métrica recall_macro) ----------------------
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring=recall_macro_scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True  # refit con el mejor
    )
    grid.fit(X_tr_pca, y_tr)

    # ---------------------- guardar resultados CV (csv por batch + acumulado) ----------------------
    cv_df = pd.DataFrame(grid.cv_results_)
    cv_df["Batch"] = title

    # etiqueta bonita para la gráfica
    cv_df["label"] = (
        "md=" + cv_df["param_max_depth"].astype(str)
        + " | lr=" + cv_df["param_learning_rate"].astype(str)
        + " | mcw=" + cv_df["param_min_child_weight"].astype(str)
    )
    all_cv_results.append(cv_df)

    # guardar CSV del batch
    csv_cv_path = os.path.join(base_path, f"xgb_cv_results_{title}.csv")
    cv_df.to_csv(csv_cv_path, index=False)

    # ---------------------- Gráfica CV: x=n_estimators, hue=label ----------------------
    plt.figure(figsize=(10,6))
    tmp = cv_df.sort_values(["param_n_estimators","param_learning_rate","param_max_depth"])
    sns.lineplot(
        data=tmp,
        x="param_n_estimators",
        y="mean_test_score",
        hue="label",
        marker="o"
    )
    plt.title(f"{title} - XGBoost CV (Recall_macro)")
    plt.xlabel("n_estimators")
    plt.ylabel("Mean CV Recall (macro)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="config", bbox_to_anchor=(1.02,1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # ---------------------- Seleccionar top-2 configs únicas ----------------------
    top2 = cv_df.sort_values("mean_test_score", ascending=False).head(2)

    # ---------------------- Evaluar top-2 en test ----------------------
    for _, row in top2.iterrows():
        params = {
            "max_depth": int(row["param_max_depth"]),
            "learning_rate": float(row["param_learning_rate"]),
            "n_estimators": int(row["param_n_estimators"]),
            "subsample": float(row["param_subsample"]),
            "colsample_bytree": float(row["param_colsample_bytree"]),
            "min_child_weight": int(row["param_min_child_weight"]),
            "gamma": float(row["param_gamma"]),
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42
        }
        if n_classes == 2:
            params.update({"objective": "binary:logistic", "eval_metric": "logloss"})
        else:
            params.update({"objective": "multi:softmax", "num_class": n_classes, "eval_metric": "mlogloss"})

        model = XGBClassifier(**params)
        model.fit(X_tr_pca, y_tr)
        y_pred = model.predict(X_te_pca)          # etiquetas (no one-hot)

        acc  = accuracy_score(y_te, y_pred)
        recm = recall_score(y_te, y_pred, average="macro")
        cm   = confusion_matrix(y_te, y_pred)

        # figura de matriz de confusión
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title(f"{title} | md={params['max_depth']} lr={params['learning_rate']} "
                  f"ne={params['n_estimators']} mcw={params['min_child_weight']}")
        plt.xticks(rotation=90)
        plt.show()

        # guardar fila de test
        all_test_results.append({
            "Batch": title,
            "max_depth": params["max_depth"],
            "learning_rate": params["learning_rate"],
            "n_estimators": params["n_estimators"],
            "min_child_weight": params["min_child_weight"],
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "gamma": params["gamma"],
            "Accuracy": acc,
            "Recall_macro": recm,
            "Confusion_Matrix": cm.tolist()
        })

        # (opcional) guardar imágenes por carpeta si ya tienes esta función
        # save_images_to_folders(base_path, title, X_te.reshape((-1,)+original_shape), y_te, y_pred, original_shape)

# ---------------------- CSVs finales (acumulados) ----------------------
cv_all   = pd.concat(all_cv_results, ignore_index=True)
test_all = pd.DataFrame(all_test_results)

cv_all.to_csv(os.path.join(base_path, "xgb_cv_results_ALL.csv"), index=False)
test_all.to_csv(os.path.join(base_path, "xgb_test_results_ALL.csv"), index=False)

print("\n✅ Guardado:")
print(" - CV por batch: xgb_cv_results_Batch_*.csv")
print(" - CV global:    xgb_cv_results_ALL.csv")
print(" - TEST global:  xgb_test_results_ALL.csv")











'''   
    
    
filtered_signal = np.convolve(signal_with_noise, np.ones(window_size) / window_size, mode='same')

# Aplicar filtro de la mediana
median_filtered_signal = medfilt(signal_with_noise, kernel_size=window_size) # Tamaño de ventana 5
# Aplicar filtro mediana 3x3
median_filtered = cv2.medianBlur(noisy_image, 3)

# Filtro Hampel
hampel_filtered_signal = hampel(signal_with_noise, window_size=window_size)


# Calcular el histograma
hist = cv2.calcHist([gray_img], [0], None, [bins], [0, 256])
# Aplanamos matriz
vector = gray_img.flatten()
# Histograma
plt.hist(vector, bins=50);


# Generar el negativo de la imagen
imagen_negativa = np.max(gray_img) - gray_img
print(imagen_negativa[600,500]) # Imprimimos un Pí­xel Gray Scale
plt.imshow(imagen_negativa, cmap='gray');

# Transformación logarí­tmica
img_normalizada = gray_img.astype(np.float32) + 1  # Añadir 1 para evitar log(0)
# Puedes ajustar este valor para escalar la salida
c_log = 255 / np.log10(1 + 255)
imagen_logaritmica = c_log * np.log10(1 + img_normalizada)
# Convertir la imagen de vuelta a formato uint8
imagen_logaritmica = np.uint8(np.clip(imagen_logaritmica, 0, 255))
plt.figure(figsize=(12, 7))
plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(gray_img, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Transformación Logarí­tmica")
plt.imshow(imagen_logaritmica, cmap='gray')
plt.show()


# Crear una matriz vací­a para la nueva imagen transformada
img2 = np.zeros((rows, columns), dtype=np.uint8)

# Parámetros de la transformación Gamma Correction
gamma = 0.08  # Valor de gamma
c = 255 / (255 ** gamma)  # Constante de normalización

# Aplicar la transformación Gamma Correction pí­xel a pí­xel
for x in range(rows):
    for y in range(columns):
        img2[x, y] = np.clip(c * (img[x, y] ** gamma), 0, 255)

plt.figure(figsize=(12, 7))
plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(gray_img, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Transformación Gamma Correction")
plt.imshow(img2, cmap='gray')
plt.show()



#Ecualizacion de histograma 
# Aplicar ecualización del histograma
equalized_image = cv2.equalizeHist(gray_img)

# Configurar la figura y tamaño
plt.figure(figsize=(14, 10))

# Imagen original
plt.subplot(2, 2, 1)
plt.title("Original")
plt.imshow(gray_img, cmap='gray')
plt.axis('off')  # Eliminar ejes

# Imagen ecualizada
plt.subplot(2, 2, 2)
plt.title("Ecualizada")
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')  # Eliminar ejes

# Lo que se hace adicionalmente:
# Cálculo manual del histograma con:
hist_o, bins_o = np.histogram(gray_img.flatten(), 256, [0,256])
# Esto genera la distribución de frecuencias para cada nivel de gris.

# Cálculo de la CDF (función de distribución acumulativa):
cdf_o = hist_o.cumsum()
cdf_normalized_o = cdf_o * float(hist_o.max()) / cdf_o.max()
# Esto ayuda a entender cómo se estira o redistribuye el rango de intensidades.


# 2. Crear un objeto CLAHE y aplicar
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# clipLimit=2.0: Lí­mite de contraste; evita que el histograma se sobreamplifique y se genere ruido.
# tileGridSize=(8, 8): Divide la imagen en 8í8 bloques y aplica ecualización localmente, luego interpola entre ellos.
img_clahe = clahe.apply(img)


# El filtro gaussiano es muy íºtil para suavizar imágenes y reducir ruido antes de aplicar tí©cnicas más sensibles, 
# como detección de bordes o extracción de caracterí­sticas
smoothed = cv2.GaussianBlur(gray_img, (5, 5), 1.0)
# Parámetro	Quí© significa	                                Recomendación
# ksize	    Tamaño del kernel (debe ser impar: 3, 5, 7...)	Más grande â más suavizado (pero pierde detalle)
# sigma	    Desviación estándar de la campana gaussiana	    Controla cuánto difumina (0 = auto por OpenCV)


# filtro bilateral
# Es un suavizado no lineal que preserva los bordes al suavizar la imagen, porque:
# Considera distancia espacial entre pí­xeles (como el gaussiano).
# Pero tambií©n diferencia de intensidad (a diferencia del gaussiano).
# Esto evita que se mezclen regiones con bordes fuertes, como el borde de un tumor.
bilateral_filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
# Parámetro	    Quí© hace	                                               Recomendación
# d	            Tamaño del vecindario (diámetro del filtro, en pí­xeles)	   Usa 9 por defecto. Si d = -1, se calcula automáticamente.
# sigmaColor	Cuánto influye la diferencia de intensidad	               Mayor valor â más suavizado (difumina más colores distintos)
# sigmaSpace	Cuánto influye la distancia fí­sica entre pí­xeles	       Mayor valor â más alejados pueden influir


# Filtro de Hampel
# Es un filtro robusto para eliminar valores atí­picos (outliers) en datos. Se basa en detectar si un valor se desví­a 
# demasiado de la mediana local, usando una ventana deslizante.
# Es muy eficaz en imágenes con artefactos puntuales o ruido "raro", preservando los bordes mejor que el filtro gaussiano.
from scipy.ndimage import median_filter
import numpy as np

def hampel_filter_2d(image, window_size=3, n_sigmas=3):
    """
    Aplica un filtro de Hampel a una imagen 2D.
    - window_size: tamaño de la ventana deslizante (debe ser impar).
    - n_sigmas: níºmero de sigmas para determinar si un valor es atí­pico.
    """
    median = median_filter(image, size=window_size)
    diff = np.abs(image - median)
    mad = median_filter(diff, size=window_size)

    threshold = n_sigmas * 1.4826 * mad
    mask = diff > threshold
    filtered_image = image.copy()
    filtered_image[mask] = median[mask]
    return filtered_image.astype(np.uint8)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Asegíºrate de estar en escala de grises
filtered = hampel_filter_2d(gray_img, window_size=3, n_sigmas=2)
# Parámetro	    Quí© hace	                                                    Recomendación
# window_size	Tamaño de la ventana (3x3, 5x5, etc.)	                        3 o 5 está bien
# n_sigmas	    Umbral de sensibilidad (outliers = valores lejanos a mediana)	2 o 3 es razonable
               
'''