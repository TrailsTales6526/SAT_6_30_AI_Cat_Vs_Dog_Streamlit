from fastai.vision.all import *
path = untar_data(URLs.PETS)
path.ls()
files = get_image_files(path/"images")
len(files)

def label_function(file_name):
  if file_name[0].isupper():
    return True
  else:
    return False

img = PILImage.create(files[2987])
img.show()

dls = ImageDataLoaders.from_name_func(path, files, label_function, item_tfms=Resize(224))

dls.show_batch()

learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)

from google.colab import files
uploaded_file = files.upload()

img_content = None
for file_name in uploaded_file.keys():
    img_content = uploaded_file[file_name]

learn.predict(img_content)