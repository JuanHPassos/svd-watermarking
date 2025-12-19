# SVD Image Watermarking 

Este projeto implementa uma técnica de marca d'água digital (watermarking) utilizando a **Decomposição em Valores Singulares (SVD)** processada em blocos de $32 \times 32$.

O objetivo é esconder uma imagem dentro de outra de forma imperceptível, utilizando álgebra linear para garantir que a informação possa ser extraída posteriormente.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JuanHPassos/svd-watermarking/blob/main/main.ipynb)

## Como funciona
1. **Embedding:** A imagem principal é dividida em blocos e decomposta via SVD. A marca d'água é inserida nos valores singulares ($\Sigma$).
2. **Extração:** Utilizando a imagem original e as chaves geradas, o algoritmo recupera a marca d'água escondida.

## Como testar
O código foi projetado para rodar no **Google Colab**. Para testar:

1. Clique no botão **Open in Colab** acima.
2. Certifique-se de que as imagens de teste (`photographer.jpg` e `lock.jpg`) estejam na mesma pasta ou faça o upload delas na aba lateral do Colab.
3. As bibliotecas necessárias são:
   * `numpy`
   * `opencv-python`
   * `google.colab`

---
*Dica: Você pode ajustar a variável `ALPHA` no código para testar a visibilidade da marca d'água.*
