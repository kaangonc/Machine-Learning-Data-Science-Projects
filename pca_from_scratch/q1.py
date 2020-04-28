
import numpy as np
from PIL import Image

def read_images():
  path = "cats/"
  images = []
  for i in range(5000):
    pic = path + str((i+1)) + ".jpg"
    img = Image.open(pic)
    img = np.array(img)
    images.append(np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2])))
    print(i)
  return np.array(images)


cats = read_images()

def pca(x):
  means = np.mean(x, axis=0)
  centered = x - means
  cov = np.cov(centered.T)
  eigen_values, eigen_vectors = np.linalg.eigh(cov)
  print(eigen_vectors.shape)
  proportions = np.divide(eigen_values, np.sum(eigen_values))
  sorted_indices = np.argsort(eigen_values)
  sorted_indices = np.flip(sorted_indices)
  return eigen_vectors, proportions, sorted_indices


def create_lists(cats):
  principal_components_list = []
  proportions_list = []
  sorted_indices_list = []
  for i in range(3):
    principal_components, proportions, sorted_indices = pca(cats[:,:,i])
    principal_components_list.append(principal_components)
    proportions_list.append(proportions)
    sorted_indices_list.append(sorted_indices)

  return np.array(principal_components_list), np.array(proportions_list), np.array(sorted_indices_list)

principal_components_list, proportions_list, sorted_indices_list = create_lists(cats)


def perform_1_1(proportions_list, sorted_indices_list):
  print(proportions_list.shape)
  for i in range(3):
    print("For channel", i)
    for j in range(10):
      print(proportions_list[i, sorted_indices_list[i][j]])
      
perform_1_1(proportions_list, sorted_indices_list)

def perform_1_2(principal_components_list, sorted_indices_list):
  path = "pc_cats/"
  new_images= np.zeros((10,64,64,3))
  for i in range(3):
    count = 0
    for j in sorted_indices_list[i,:10]:
      principal_component = principal_components_list[i][:,j]
      principal_component = np.reshape(principal_component, (64,64))
      new_images[count,:,:,i] = principal_component
      count += 1

  for i in range(10):
    Image.fromarray((255*new_images[i]).astype(np.uint8)).save(path + 'pc'+str(i+1)+'.jpg')

perform_1_2(principal_components_list, sorted_indices_list)

def perform_1_3(cats, principal_components_list, sorted_indices_list):
  path = "reconstructed_cats/"
  k_values = [1,50,250,500]
  reconstructs = np.zeros((4,64,64,3))
  for i in range(3):
    first = cats[0,:,i]
    count_k = 0
    for k in k_values:
      k_principal_components = np.zeros((k,4096))
      for k_ in range(k):
        k_principal_components[k_,:] = principal_components_list[i,:,sorted_indices_list[i, k_]]
      reconstruct = np.dot(first, k_principal_components.T)
      reconstruct = np.dot(reconstruct, k_principal_components)
      reconstructs[count_k,:,:,i] = np.reshape(reconstruct, (64,64))
      count_k += 1
  
  for i in range(len(k_values)):
    Image.fromarray((reconstructs[i]).astype(np.uint8)).save(path + 'reconstructedBy'+str(k_values[i])+'.jpg')

perform_1_3(cats, principal_components_list, sorted_indices_list)