# SVD Image Watermarking 

Este projeto implementa uma técnica de marca d'água digital (watermarking) utilizando a **Decomposição em Valores Singulares (SVD)** processada em blocos de $32 \times 32$.

O objetivo é esconder uma imagem dentro de outra de forma imperceptível, utilizando álgebra linear para garantir que a informação possa ser extraída posteriormente.

## Como funciona
1. **Embedding:** A imagem principal é dividida em blocos e decomposta via SVD. A marca d'água é inserida nos valores singulares ($\Sigma$).
2. **Extração:** Utilizando a imagem original e as chaves geradas, o algoritmo recupera a marca d'água escondida.



## Como testar
O projeto foi desenvolvido para ser executado no **Google Colab**.

1. Acesse o arquivo: **[SVD_Watermark.ipynb](main.ipynb)**.
2. Certifique-se de ter as imagens `photographer.jpg` e `lock.jpg` no mesmo diretório (ou faça o upload para o Colab).
3. Execute as células em ordem.

## Requisitos
Para rodar este notebook, são utilizadas as seguintes bibliotecas:
* `numpy`
* `opencv-python` (cv2)
* `google.colab` (para exibição de imagens)

---
*Dica: Você pode ajustar a variável `ALPHA` no código para testar a visibilidade da marca d'água.*
