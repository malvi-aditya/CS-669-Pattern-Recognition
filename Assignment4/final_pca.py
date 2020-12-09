import os
import numpy as np
import cv2 as cv
from PIL import Image
from skimage.util import random_noise


result_dir='answer/'


def get_vector_list(digit):
    file=[]
    for fname in os.listdir("data/"+str(digit)):
        image=cv.imread("data/"+str(digit)+str("/")+fname)
        image=cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image=image.flatten()
        file.append(image)
    return file



print("Enter number of principal components :")
number_of_pc=int(input())

print("Choose digit")
digit=int(input())

image_vector_list=get_vector_list(digit)




def add_noise(image_vector):
    for i in range(len(image_vector)):
        image_vector[i] = random_noise(image_vector[i],mode='gaussian',mean=0,var=0.15)
        image_vector[i] = np.array(255*image_vector[i])
    return image_vector


mean_vector=np.mean(image_vector_list, axis=0)
var_vector=np.var(image_vector_list,axis=0)

for x in image_vector_list:
        for i in range(len(mean_vector)):
                x[i]=x[i]-mean_vector[i]

i=0
for image in image_vector_list:
    image=np.uint8(np.clip(image,0,255))
    image_matrix=np.reshape(image,(28, 28))
    img = Image.fromarray(image_matrix, 'L')
    img.save("initial/"+str(i)+".jpg")
    i=i+1
                
                
print("Do you want to add noise(y or n) :")
noise=(input())
if noise=="y":
    image_vector_list = add_noise(image_vector_list)
    
for file in os.listdir("noise"):
    if file.endswith('.jpg'):
        os.remove("noise"+"/"+file)

i=0
for recon in image_vector_list:
    recon=np.uint8(np.clip(recon,0,255))
    image_matrix=np.reshape(recon,(28, 28))
    img = Image.fromarray(image_matrix, 'L')
    img.save("noise/"+str(i)+".jpg")
    i=i+1




convarience_matrix=np.cov(np.array(image_vector_list).T)
eig_vals, eig_vecs = np.linalg.eig(np.array(convarience_matrix))
eig_vals=np.real(eig_vals)
eig_vecs=np.real(eig_vecs)
print("Eigenvector and Eigenvalue shapes are",eig_vecs.shape, eig_vals.shape)

temp = []
for i in range(len(eig_vals)):
    temp.append([eig_vals[i],eig_vecs[:,i]])

temp.sort(key = lambda x:x[0], reverse=True)

eig_vecs = []
for i in range(number_of_pc):
    eig_vecs.append(temp[i][1])

eig_vecs = np.asarray(eig_vecs)

for file in os.listdir(result_dir):
    if file.endswith('.jpg'):
        os.remove(result_dir+"/"+file)

reconstructed=[]

print(np.asarray(eig_vecs).shape)
print(np.array(image_vector_list).shape)
for x in image_vector_list:
    projected=np.dot(eig_vecs,np.array(x))
    recon = np.dot(np.transpose(eig_vecs), projected)
    reconstructed.append(recon)

for i in range(len(reconstructed)):
    reconstructed[i]=reconstructed[i]+mean_vector

i=0
for recon in reconstructed:
    recon=np.uint8(np.clip(recon,0,255))
    image_matrix=np.reshape(recon,(28, 28))
    img = Image.fromarray(image_matrix, 'L')
    img.save("answer/"+str(i)+".jpg")
    i=i+1