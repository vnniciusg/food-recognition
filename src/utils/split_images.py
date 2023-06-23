import os
import shutil


def splitImages(images_folder, train_folder, test_folder):
    train_ratio = 0.8
    # 1- Percorrer cada pasta no diretorio de imagens
    for folder in os.listdir(images_folder):
        folder_dir = os.path.join(images_folder, folder)

        # 2- Verifica se é um diretorio
        if os.path.isdir(folder_dir):
            # 3 - Lista todas as imagens da pasta
            images = os.listdir(folder_dir)

            # 4- Calcula o númnero de imagens para treinamento , sendo a divisao 80 para treinamento e 20 para treino ( padrão )
            num_train = int(train_ratio * len(images))

            # 5 - Divide as imagens em treinamento e teste
            train_images = images[:num_train]
            test_images = images[num_train:]

            # 6 - Cria os diretorios de destino para treinamento e teste
            train_dir = os.path.join(train_folder, folder)
            test_dir = os.path.join(test_folder, folder)

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            # 7 - Move as imagens para os diretorios correspondentes
            for image in train_images:
                src_path = os.path.join(folder_dir, image)
                dst_path = os.path.join(train_dir, image)
                shutil.copy(src_path, dst_path)

            for image in test_images:
                src_path = os.path.join(folder_dir, image)
                dst_path = os.path.join(test_dir, image)
                shutil.copy(src_path, dst_path)

    print("Divisao entre treinamento e teste concluida")
