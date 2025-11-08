import numpy as np
import cv2

def embed_svd(host_image, watermark_image, alpha, block_size=4):
    """
    Embute uma marca d'água (watermark_image) em uma imagem hospedeira (host_image)
    usando SVD em blocos. 
    
    :param host_image: A imagem original (tons de cinza).
    :param watermark_image: A imagem da marca d'água (tons de cinza).
    :param alpha: O fator de força (quão forte a marca d'água é embutida).
    :param block_size: O tamanho de cada bloco (ex: 8 para 8x8).
    :return: A imagem marcada (watermarked_image).
    """
    print(f"Iniciando embedding SVD em blocos de {block_size}x{block_size}...")
    
    # 1. Converter imagens para float para cálculos
    host_image_f = host_image.astype(float)
    
    # 2. Redimensionar a marca d'água para o tamanho da hospedeira
    #    Isso simplifica o mapeamento 1:1 dos blocos.
    watermark_image_f = cv2.resize(watermark_image, (host_image.shape[1], host_image.shape[0])).astype(float)
    
    # 3. Preparar a imagem de saída
    watermarked_image = np.zeros_like(host_image_f)
    
    rows, cols = host_image.shape
    
    # 4. Iterar sobre a imagem em blocos
    for r in range(0, rows, block_size):
        for c in range(0, cols, block_size):
            
            # --- Definir os limites do bloco atual ---
            # (Lida com imagens que não são perfeitamente divisíveis pelo block_size)
            r_end = min(r + block_size, rows)
            c_end = min(c + block_size, cols)
            
            # 5. Obter os blocos da hospedeira e da marca d'água
            host_block = host_image_f[r:r_end, c:c_end]
            watermark_block = watermark_image_f[r:r_end, c:c_end]
            
            # Salvar o formato real do bloco (pode ser menor nas bordas)
            current_block_shape = host_block.shape
            
            # 6. Aplicar SVD no bloco da IMAGEM HOSPEDEIRA
            try:
                U_h, S_h_diag, V_h_T = np.linalg.svd(host_block)
            except np.linalg.LinAlgError:
                # Se SVD falhar (ex: bloco com todos os pixels iguais), 
                # apenas copie o bloco original.
                watermarked_image[r:r_end, c:c_end] = host_block
                continue

            # 7. Reconstruir S_h como uma matriz diagonal
            S_h = np.zeros(current_block_shape, dtype=float)
            diag_len = S_h_diag.shape[0]
            S_h[:diag_len, :diag_len] = np.diag(S_h_diag)

            # 8. Adicionar a marca d'água (fórmula D = S + a*W) [cite: 147]
            #    Note que S_h e watermark_block devem ter o mesmo tamanho.
            S_w_block = S_h + alpha * watermark_block
            
            # 9. Reconstruir o bloco marcado
            #    Imagem_marcada = U_h * S_w * V_h_T
            watermarked_block = U_h @ S_w_block @ V_h_T
            
            # 10. Colocar o bloco processado na imagem de saída
            watermarked_image[r:r_end, c:c_end] = watermarked_block

    # 11. Clampar/Truncar valores e converter de volta para uint8
    watermarked_image = np.clip(watermarked_image, 0, 255)
    watermarked_image = np.uint8(watermarked_image)
    
    print("Embedding em blocos concluído.")
    return watermarked_image


def extract_svd(watermarked_image, host_image, alpha, block_size=4):
    """
    Extrai uma marca d'água (watermark_image) de uma imagem marcada (watermarked_image)
    usando SVD em blocos.
    
    NOTA: Este é um método "não-cego" (non-blind) e REQUER a imagem hospedeira
    original para a extração, conforme descrito na fórmula do artigo. [cite: 222]
    
    :param watermarked_image: A imagem marcada.
    :param host_image: A imagem hospedeira ORIGINAL.
    :param alpha: O fator de força usado no embedding.
    :param block_size: O tamanho de bloco usado no embedding.
    :return: A imagem da marca d'água extraída.
    """
    print(f"Iniciando extração SVD em blocos de {block_size}x{block_size}...")

    # 1. Converter imagens para float
    watermarked_image_f = watermarked_image.astype(float)
    host_image_f = host_image.astype(float)

    # 2. Preparar a imagem de saída para a marca d'água
    extracted_watermark = np.zeros_like(host_image_f)
    
    rows, cols = host_image.shape

    # 3. Iterar sobre a imagem em blocos (deve ser idêntico ao embedding)
    for r in range(0, rows, block_size):
        for c in range(0, cols, block_size):
            
            # --- Definir os limites do bloco atual ---
            r_end = min(r + block_size, rows)
            c_end = min(c + block_size, cols)
            
            # 4. Obter os blocos da imagem marcada e da hospedeira
            watermarked_block = watermarked_image_f[r:r_end, c:c_end]
            host_block = host_image_f[r:r_end, c:c_end]
            
            current_block_shape = host_block.shape

            # 5. Calcular SVD de AMBOS os blocos para obter suas matrizes S
            
            try:
                # --- Obter S da hospedeira (S) ---
                _, S_h_diag, _ = np.linalg.svd(host_block)
                S_h = np.zeros(current_block_shape, dtype=float)
                diag_len_h = S_h_diag.shape[0]
                S_h[:diag_len_h, :diag_len_h] = np.diag(S_h_diag)

                # --- Obter S da marcada (D) ---
                _, S_w_diag, _ = np.linalg.svd(watermarked_block)
                S_w = np.zeros(current_block_shape, dtype=float) # O artigo chama isso de 'D'
                diag_len_w = S_w_diag.shape[0]
                S_w[:diag_len_w, :diag_len_w] = np.diag(S_w_diag)
                
            except np.linalg.LinAlgError:
                # SVD falhou, pular este bloco
                continue
                
            # 6. Aplicar a fórmula de extração: W = (D - S) / alpha [cite: 222]
            #    (Onde D é S_w e S é S_h)
            watermark_block = (S_w - S_h) / alpha
            
            # 7. Colocar o bloco da marca d'água extraído na imagem de saída
            extracted_watermark[r:r_end, c:c_end] = watermark_block

    # 8. Clampar/Truncar valores e converter
    extracted_watermark = np.clip(extracted_watermark, 0, 255)
    extracted_watermark = np.uint8(extracted_watermark)
    
    print("Extração em blocos concluída.")
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