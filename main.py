# pip install tensorflow
# pip install scikit-learn
# pip install pillow
# pip install matplotlib

import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import os
import matplotlib.pyplot as plt

# Carregar o modelo VGG16
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features

def exibir_imagens(imagem_referencia, imagens_selecionadas):
    img_ref = image.load_img(imagem_referencia)
    plt.figure(figsize=(10, 2))

    # Mostrar a imagem de referência
    plt.subplot(1, len(imagens_selecionadas) + 1, 1)
    plt.imshow(img_ref)
    plt.title('Referência')
    plt.axis('off')

    # Mostrar as imagens selecionadas
    for i, img_path in enumerate(imagens_selecionadas):
        img = image.load_img(img_path)
        plt.subplot(1, len(imagens_selecionadas) + 1, i + 2)
        plt.imshow(img)
        plt.title(f'Recomendação {i + 1}')
        plt.axis('off')

    plt.show()

# Caminho da imagem de entrada
input_image_path = 'sofa_ref.jpg'

# Diretório das imagens a serem analisadas
images_dir = './images'

# Extrair características da imagem de entrada
input_features = extract_features(input_image_path, model)

# Extrair características das imagens no diretório
image_paths = [os.path.join(images_dir, img) for img in os.listdir(images_dir)]
features = np.zeros((len(image_paths), 4096))

for i, img_path in enumerate(image_paths):
    features[i, :] = extract_features(img_path, model)

# Calcular a similaridade entre as imagens
similarity = cosine_similarity(input_features, features)

# Obter as imagens mais semelhantes
similar_images_indices = similarity[0].argsort()[-4:][::-1]

# Caminhos das imagens selecionadas
similar_images = [image_paths[i] for i in similar_images_indices]

# Exibir as imagens
exibir_imagens(input_image_path, similar_images)