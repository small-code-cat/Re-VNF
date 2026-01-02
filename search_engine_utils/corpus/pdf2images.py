import os
from tqdm import tqdm
from pdf2image import convert_from_path

root_path = '/data/user0/PycharmProjects/search_engine_MMDocRAG/corpus'
os.makedirs(os.path.join(root_path, 'img'), exist_ok=True)
pdf_path = '/data/user0/datasets/MMDocIR/MMDocRAG/doc_pdfs'
pdf_files = [file for file in os.listdir(pdf_path) if file.endswith('pdf')]
for filename in tqdm(pdf_files):
    filepath = os.path.join(pdf_path,filename)
    imgname = filename.split('.pdf')[0]
    images = convert_from_path(filepath)
    for i, image in enumerate(images):
        idx = i + 1
        img_path = os.path.join(root_path, 'img', f'{imgname}_{idx}.jpg')
        if not os.path.exists(img_path):
            image.save(img_path, 'JPEG')
            print(f'The image is saved to {img_path}')