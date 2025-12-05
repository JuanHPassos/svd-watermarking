import numpy as np
from PIL import Image
import random

def load_and_scale_images(image_path, watermark_path):
    image = Image.open(image_path).convert('L')  
    image = image.resize((512, 512))
    image_array = np.array(image, dtype=np.float64)
    
    watermark = Image.open(watermark_path).convert('L')
    watermark = watermark.resize((32, 32))
    watermark_array = np.array(watermark) > 128  
    
    return image_array, watermark_array, image, watermark

img_array, wtr_array, img, wtr = load_and_scale_images('photographer.jpg', 'lock.jpg')
wtr.save('watermark_32x32.png')

print(f"Watermark original - Bits 1: {np.sum(wtr_array)}, Bits 0: {np.sum(~wtr_array)}")

def svd_decomposition(image_array, block_size=8):
    blocks = []
    original_indices = []
    complexity_scores = []
    
    for i in range(0, image_array.shape[0], block_size):
        for j in range(0, image_array.shape[1], block_size):
            block = image_array[i:i+block_size, j:j+block_size]
            U, D, Vt = np.linalg.svd(block, full_matrices=False)
            complexity_score = np.sum(D ** 2)
            
            blocks.append((U.copy(), D.copy(), Vt.copy()))
            original_indices.append((i, j))
            complexity_scores.append(complexity_score)
    
    # Determinar limiar de 70º percentil
    threshold = np.percentile(complexity_scores, 70)
    
    # Extrair índices de blocos complexos
    complex_indices = [idx for idx in range(len(blocks)) 
                      if complexity_scores[idx] >= threshold]
    
    return complex_indices, blocks, original_indices

complex_indices, all_blocks, all_indices = svd_decomposition(img_array)
print(f"Número de blocos complexos: {len(complex_indices)}")
print(f"Total de blocos: {len(all_blocks)}")

def embed_watermark_in_blocks(complex_indices, all_blocks, watermark_array, seed=42, alpha=0.15):
    # CRUCIAL: Usar índice sequencial ao invés de random.choice
    # Isso garante que cada bit use um bloco diferente
    random.seed(seed)
    random.shuffle(complex_indices)  # Embaralhar uma vez
    
    watermark_bits = watermark_array.flatten()
    num_bits = len(watermark_bits)
    
    if num_bits > len(complex_indices):
        print(f"AVISO: Não há blocos complexos suficientes! Necessário: {num_bits}, Disponível: {len(complex_indices)}")
        return all_blocks, []
    
    print(f"\nEmbutindo {num_bits} bits em blocos únicos...")
    
    block_selection = []
    bits_1_count = 0
    bits_0_count = 0
    
    # Cada bit usa um bloco DIFERENTE
    for bit_idx, bit in enumerate(watermark_bits):
        selected_idx = complex_indices[bit_idx]  # Uso sequencial, não aleatório!
        block_selection.append(selected_idx)
        
        U, D, Vt = all_blocks[selected_idx]
        
        # Salvar valores originais
        orig_u00 = U[0, 0]
        orig_u10 = U[1, 0]
        
        # Modificar U[0,0] e U[1,0] baseado no bit com margem maior
        if bit == 1:
            bits_1_count += 1
            # Forçar U[0,0] > U[1,0]
            mid = (U[0, 0] + U[1, 0]) / 2
            U[0, 0] = mid + alpha
            U[1, 0] = mid - alpha
        else:
            bits_0_count += 1
            # Forçar U[0,0] < U[1,0]
            mid = (U[0, 0] + U[1, 0]) / 2
            U[0, 0] = mid - alpha
            U[1, 0] = mid + alpha
        
        # Debug
        if bit_idx < 5 or bit_idx >= num_bits - 5:
            print(f"Bit {bit_idx} = {bit} (bloco {selected_idx}): U[0,0]: {orig_u00:.4f} -> {U[0,0]:.4f}, U[1,0]: {orig_u10:.4f} -> {U[1,0]:.4f}, diff={U[0,0]-U[1,0]:.4f}")
        
        all_blocks[selected_idx] = (U, D, Vt)
    
    print(f"Bits 1 embutidos: {bits_1_count}, Bits 0 embutidos: {bits_0_count}")
    
    return all_blocks, block_selection

def reconstruct_image(blocks, indices, block_size=8):
    reconstructed_image = np.zeros((512, 512))
    
    for (U, D, Vt), (i, j) in zip(blocks, indices):
        reconstructed_block = np.dot(U, np.dot(np.diag(D), Vt))
        reconstructed_image[i:i+block_size, j:j+block_size] = reconstructed_block
    
    return reconstructed_image

# Embutir marca d'água
watermarked_blocks, embedding_selection = embed_watermark_in_blocks(
    complex_indices.copy(), all_blocks, wtr_array
)

# Reconstruir imagem
reconstructed_image = reconstruct_image(watermarked_blocks, all_indices)
reconstructed_image = np.clip(reconstructed_image, 0, 255)

print(f"\nForma da imagem reconstruída: {reconstructed_image.shape}")

new_image = Image.fromarray(np.uint8(reconstructed_image))
new_image.save('photographer_watermarked.png')
print("Imagem com watermark salva: 'photographer_watermarked.png'")

def extract_watermark_from_blocks(watermarked_blocks, block_selection, 
                                 watermark_size=(32, 32)):
    extracted_bits = []
    num_bits = watermark_size[0] * watermark_size[1]
    
    print(f"\nExtraindo {num_bits} bits dos mesmos blocos...")
    
    extracted_1_count = 0
    extracted_0_count = 0
    
    # Usar EXATAMENTE os mesmos blocos na mesma ordem
    for bit_idx in range(num_bits):
        selected_idx = block_selection[bit_idx]
        
        U, D, Vt = watermarked_blocks[selected_idx]
        
        # Extrair bit baseado na comparação
        if U[0, 0] > U[1, 0]:
            extracted_bits.append(1)
            extracted_1_count += 1
        else:
            extracted_bits.append(0)
            extracted_0_count += 1
        
        # Debug
        if bit_idx < 5 or bit_idx >= num_bits - 5:
            print(f"Bit {bit_idx} (bloco {selected_idx}): U[0,0] = {U[0,0]:.4f}, U[1,0] = {U[1,0]:.4f}, diff={U[0,0]-U[1,0]:.4f} -> {extracted_bits[-1]}")
    
    print(f"Bits 1 extraídos: {extracted_1_count}, Bits 0 extraídos: {extracted_0_count}")
    
    extracted_watermark = np.array(extracted_bits).reshape(watermark_size)
    return extracted_watermark

# Extrair marca d'água usando os MESMOS blocos
extracted_watermark = extract_watermark_from_blocks(
    watermarked_blocks, embedding_selection
)

extracted_image = Image.fromarray(np.uint8(extracted_watermark * 255))
extracted_image.save('extracted_watermark.png')
print("Watermark extraído salvo: 'extracted_watermark.png'")

# Verificações
print(f"\n=== VERIFICAÇÕES ===")

# Comparar watermark original com extraído
correct_bits = np.sum(wtr_array == extracted_watermark)
total_bits = wtr_array.size
accuracy = correct_bits / total_bits * 100

print(f"Bits corretos: {correct_bits}/{total_bits}")
print(f"Acurácia da extração: {accuracy:.2f}%")

# Mostrar estatísticas detalhadas
print(f"\n=== ESTATÍSTICAS DETALHADAS ===")
print(f"Original - 1s: {np.sum(wtr_array)}, 0s: {np.sum(~wtr_array)}")
print(f"Extraído - 1s: {np.sum(extracted_watermark)}, 0s: {np.sum(~extracted_watermark)}")

# Verificar se há blocos duplicados
if len(embedding_selection) != len(set(embedding_selection)):
    duplicates = len(embedding_selection) - len(set(embedding_selection))
    print(f"⚠️  AVISO: {duplicates} blocos foram usados mais de uma vez!")
else:
    print(f"✅ Todos os {len(embedding_selection)} blocos são únicos")

# Criar comparação visual
comparison_width = 32 * 3 + 20
comparison_height = 32

comparison = Image.new('L', (comparison_width, comparison_height), 255)

original_img = Image.fromarray(np.uint8(wtr_array * 255))
comparison.paste(original_img, (0, 0))

extracted_img = Image.fromarray(np.uint8(extracted_watermark * 255))
comparison.paste(extracted_img, (42, 0))

diff = np.abs(wtr_array.astype(int) - extracted_watermark.astype(int))
diff_img = Image.fromarray(np.uint8((1 - diff) * 255))
comparison.paste(diff_img, (84, 0))

comparison = comparison.resize((comparison_width * 8, comparison_height * 8), Image.NEAREST)
comparison.save('watermark_comparison.png')
print(f"\nComparação salva em 'watermark_comparison.png'")
print("(Esquerda: Original | Centro: Extraído | Direita: Diferenças - Branco=Correto)")
print("\n✅ Processo concluído! Verifique as imagens geradas.")