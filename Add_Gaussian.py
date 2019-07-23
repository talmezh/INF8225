import numpy as np
import os
#import scipy.misc
import pandas as pd


def label2MaskMap(data, c_dx = 0, c_dy = 0, shape=(512,512), radius = 10, normalize = False):
    """
    Generate a Mask map from the coordenates
    :param M, N: dimesion of output
    :param position: position of the label
    :param radius: is the radius of the gaussian function
    :return: a MxN normalized array
    """

    # Our 2-dimensional distribution will be over variables X and Y
    (M,N) = (shape[0], shape[1])
    #if len(data)<=2:
    #    data = [data]

    maskMap = []
    for index, value in enumerate(data):
        x,y = value

        #Correct the labels
        x = x + c_dx
        y = y + c_dy

        X = np.linspace(0, M - 1, M)
        Y = np.linspace(0, N - 1, N)
        X, Y = np.meshgrid(X, Y)
        # Pack X and Y into a single 3-dimensional array
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        # Mean vector and covariance matrix
        mu = np.array([x, y])
        Sigma = np.array([[radius, 0], [0, radius]])

        # The distribution on the variables X, Y packed into pos.
        Z = multivariate_gaussian(pos, mu, Sigma)

        # Normalization
        if normalize:
            Z = Z * (1 / np.max(Z))
        else:
            # 8bit image values (the loss go to inf+)
            Z = Z * (1 / np.max(Z))
            Z = np.asarray(Z * 255, dtype=np.uint8)

        maskMap.append(Z)

    if len(maskMap) == 1:
        maskMap = maskMap[0]

    return np.asarray(maskMap)

def multivariate_gaussian(pos, mu, Sigma):
    """
    Return the multivariate Gaussian distribution on array.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

if __name__ == "__main__":
    #Annotation file directory
    annotation_dir = "/home/talmezh/Documents/Data/CIL-01/Annotation"
    
    
    #Find all annotation files in directory
    files = [file for file in os.listdir(annotation_dir) if file.endswith('txt')]
    
    
    for target_nb, file in enumerate(files):
        
        save_dir = annotation_dir + "/Output"+ str(target_nb)
        if os.path.isdir(save_dir) != True:
            os.mkdir(save_dir)
        #Load annotation coordinates as pandas DataFrame
        annot = pd.read_csv(os.path.join(annotation_dir,file),
                            sep="   ",
                            header=None)
        annot.columns = ["frame", "x", "y"]
        
        #Generate gaussian distribution and save distributions as .npy binaries
    
        for index, frame in enumerate(annot.frame):
            
            mask = label2MaskMap([(annot.x[index],annot.y[index])], 
                                  shape = (640,480),
                                  radius = 100, 
                                  normalize = True)
            
            np.save(os.path.join(save_dir, "{:0>4d}".format(int(frame))),mask)
            
            #Uncomment to save as JPEG file
            #scipy.misc.toimage(mask, cmin=0.0, cmax=1).save(os.path.join(save_dir,
            #                  "{:0>4d}".format(int(frame))+'.jpg'))
            
