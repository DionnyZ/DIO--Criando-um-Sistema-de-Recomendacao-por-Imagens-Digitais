# Criando um Sistema de Recomendação por Imagens Digitais

O objetivo do projeto era criar um aplicativo que fizesse uma busca e recomendasse imagens semelhantes a uma outra imagem de entrada. A busca era para ser feita na internet, mas os métodos que encontrei precisavam utilizar serviços em nuvem como AWS, Azure e Google Cloud Platform, o que poderia gerar custos. Para evitar isso, adaptei o projeto para fazer a análise de imagens baixadas previamente.

Para testes, foram utilizadas uma imagem de entrada e uma pasta com outras trinta. O aplicativo realiza a comparação da imagem de entrada com as demais, seleciona as quatro mais semelhantes e as exibe utilizando o Matplotlib.

## Resultado

<img src="resultado.jpg" width="800">
