# Versão sem divisão por blocos

import numpy as np
import cv2
import warnings

def embed_svd(host_image, watermark_image, alpha):
    """
    Embute uma marca d'água (watermark_image) em uma imagem hospedeira (host_image)
    usando SVD.
    
    :param host_image: A imagem original (tons de cinza).
    :param watermark_image: A imagem da marca d'água (tons de cinza).
    :param alpha: O fator de força (quão forte a marca d'água é embutida).
    :return: A imagem marcada (watermarked_image).
    """
    
    print("Iniciando processo de embedding...")

    # 1. Redimensionar a marca d'água para o tamanho da imagem hospedeira
    #    O SVD requer que as matrizes tenham dimensões compatíveis para a soma.
    watermark_image_resized = cv2.resize(watermark_image, (host_image.shape[1], host_image.shape[0]))
    
    # 2. Aplicar SVD na imagem HOSPEDEIRA
    #    U_h, S_h, V_h_T = U (vetores singulares esquerdos), S (valores singulares), V Transposto (vetores singulares direitos)
    U_h, S_h_diag, V_h_T = np.linalg.svd(host_image)
    
    #    np.linalg.svd retorna S_h_diag como um vetor 1D. Precisamos convertê-lo em uma
    #    matriz diagonal (S_h) com o mesmo tamanho da imagem hospedeira.
    S_h = np.zeros_like(host_image, dtype='float')
    S_h[:S_h_diag.shape[0], :S_h_diag.shape[0]] = np.diag(S_h_diag)

    # 3. Adicionar a marca d'água aos valores singulares da hospedeira
    #    Esta é a fórmula principal do embedding: S_marcada = S_hospedeira + alpha * marca_d'água
    #    Usamos a marca d'água redimensionada diretamente.
    S_w = S_h + alpha * watermark_image_resized

    # 4. Reconstruir a imagem marcada
    #    Usamos os U e V originais da imagem hospedeira, mas a *nova* matriz S (S_w).
    #    Imagem_marcada = U_h * S_w * V_h_T
    watermarked_image = U_h @ S_w @ V_h_T
    
    # 5. Clampar/Truncar valores de pixel
    #    A operação matemática pode criar pixels > 255 (brancos) ou < 0 (pretos).
    #    Nós "cortamos" esses valores para garantir que fiquem no intervalo [0, 255].
    watermarked_image = np.clip(watermarked_image, 0, 255)
    
    # Converter de volta para uint8 (tipo de imagem padrão)
    watermarked_image = np.uint8(watermarked_image)
    
    print("Embedding concluído.")
    return watermarked_image

def extract_svd(watermarked_image, host_image, alpha):
    """
    Extrai uma marca d'água de uma imagem marcada (watermarked_image).
    Este é um método "não-cego" (non-blind), pois precisa da imagem hospedeira original.
    
    :param watermarked_image: A imagem que contém a marca d'água.
    :param host_image: A imagem hospedeira original (necessária para a extração).
    :param alpha: O mesmo fator de força usado no embedding.
    :return: A marca d'água extraída (extracted_watermark).
    """
    
    print("Iniciando processo de extração...")
    
    # Ignorar avisos de tipo (comum em operações SVD)
    warnings.filterwarnings('ignore')

    # 1. Aplicar SVD na imagem MARCADA
    U_w, S_w_diag, V_w_T = np.linalg.svd(watermarked_image)
    S_w = np.zeros_like(watermarked_image, dtype='float')
    S_w[:S_w_diag.shape[0], :S_w_diag.shape[0]] = np.diag(S_w_diag)
    
    # 2. Aplicar SVD na imagem HOSPEDEIRA ORIGINAL
    U_h, S_h_diag, V_h_T = np.linalg.svd(host_image)
    S_h = np.zeros_like(host_image, dtype='float')
    S_h[:S_h_diag.shape[0], :S_h_diag.shape[0]] = np.diag(S_h_diag)

    # 3. Inverter a fórmula de embedding para extrair a marca d'água
    #    Fórmula de embedding: S_w = S_h + alpha * W_extraida
    #    Fórmula de extração:  W_extraida = (S_w - S_h) / alpha
    #
    #    Aqui está o "truque": como o SVD não garante que U e V sejam os mesmos
    #    para a imagem hospedeira e a marcada, não podemos simplesmente subtrair S_w - S_h.
    #    Em vez disso, revertemos a reconstrução da imagem marcada usando os U e V da hospedeira.
    
    #    Passo 3a: Decompor S_w usando os U e V da *hospedeira*.
    #    Isso isola as mudanças que a marca d'água causou.
    S_w_isolada = (U_h.T @ watermarked_image @ V_h_T.T)

    #    Passo 3b: Agora podemos usar a fórmula de extração simples
    extracted_watermark_resized = (S_w_isolada - S_h) / alpha
    
    # 4. Normalizar e converter a imagem
    #    Opcional: Normalizar para 0-255 para melhor visualização
    extracted_watermark_resized = cv2.normalize(extracted_watermark_resized, None, 0, 255, cv2.NORM_MINMAX)
    
    # Converter para uint8
    extracted_watermark = np.uint8(extracted_watermark_resized)
    
    print("Extração concluída.")
    return extracted_watermark

# --- Exemplo de como usar ---

# Fator de força da marca d'água.
# Valores maiores = mais robusto, mas mais visível.
# Valores menores = menos visível, mas menos robusto.
ALPHA = 0.05

# 1. Carregar imagens em tons de cinza (0)
host = cv2.imread('photographer.jpg', 0)
watermark = cv2.imread('lock.jpg', 0)

# 2. Embutir a marca d'água
watermarked_image = embed_svd(host, watermark, ALPHA)

# 3. Extrair a marca d'água
#    Note que precisamos da imagem 'host' original.
extracted_watermark = extract_svd(watermarked_image, host, ALPHA)

# 4. Mostrar resultados
cv2.imshow('1 - Imagem Hospedeira Original', host)
cv2.imshow('2 - Marca d\'água Original', watermark)
cv2.imshow('3 - Imagem com Marca d\'água', watermarked_image)
cv2.imshow('4 - Marca d\'água Extraída', extracted_watermark)

# Salvar a imagem marcada
cv2.imwrite('watermarked_SVD.jpg', watermarked_image)

print("\n--- Resultados ---")
print(f"Fator de força (alpha): {ALPHA}")
print("Pressione qualquer tecla para fechar as janelas.")

cv2.waitKey(0)
cv2.destroyAllWindows()