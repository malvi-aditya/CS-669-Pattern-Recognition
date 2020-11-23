# Import required Libraries
from scipy.spatial import distance
import pandas as pd, numpy as np, cv2


# read and reshape input image
im = cv2.imread('Image.jpg')
img = im.reshape(-1, 3)

# Convert first data (without coordinates) to numpy array
df_a = np.array(img)


# Image segmentation with distance
# preprocess the image for including coordinate of each pixel
width, height = len(im), len(im[1])
df_b = []
W = width/255
H = height/255
for i in range(width):
    for j in range(height):
        temp = []
        temp.append(im[i][j][0])
        temp.append(im[i][j][1])
        temp.append(im[i][j][2])
        temp.append(int(i/H))
        temp.append(int(j/W))
        df_b.append(temp)
      
# Convert second data (with coordinates) into numpy array
df_b = np.array(df_b)


# Define Kmeans algorithm function
def Kmeans_algo(arr ,k=3,max_iterations=100):
    
    if isinstance(arr, pd.DataFrame):arr = arr.values
    
    # intialize k random cluster_centre 
    idx = np.random.choice(len(arr), k, replace=False)
    cluster_centre = arr[idx, :]
    
    # find nearest centroid fo reach point
    closest_centre = np.argmin(distance.cdist(arr, cluster_centre, 'euclidean'),axis=1)
    
    # run algorithm for a max no. of iteration
    for _ in range(max_iterations):
        cluster_centre = np.vstack([arr[closest_centre==i,:].mean(axis=0) for i in range(k)])
        tmp = np.argmin(distance.cdist(arr, cluster_centre, 'euclidean'),axis=1)
        if np.array_equal(closest_centre,tmp):
            break
        closest_centre = tmp
        
    # return required values
    return closest_centre, cluster_centre


#  Perform image segmentation with different no. of cluster
for k in [2,3,4]:
    
    # get cluster_centre and closest_centre for first set of data
    closest_centre, cluster_centre = Kmeans_algo(df_a, k)
    res = []
    for i in closest_centre:
        res.append(cluster_centre[i])
    res= np.array(res)
    out = res.reshape((im.shape))
    cv2.imwrite(str(k)+".jpg", out)
    
    # get cluster_centre and closest_centre for second set of data
    closest_centre, cluster_centre = Kmeans_algo(df_b, k)
    res = []
    
    colors = []
    for i in range(k):
        b, g, r = cluster_centre[i][0], cluster_centre[i][1], cluster_centre[i][2]
        colors.append([b,g,r])
        
    for i in closest_centre:
        res.append(colors[i])
    res= np.array(res)
    out = res.reshape((im.shape))
    cv2.imwrite(str(k)+"coord.jpg", out)