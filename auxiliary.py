##############################################################
###### This code defines the PH landmark selection function and all its help functions
# Original source: https://github.com/stolzbernadette/Outlier-robust-subsampling-techniques-for-persistent-homology


import numpy as np
from ripser import ripser
from scipy.spatial import KDTree


def getMaxPersistence(ripser_pd):

	if ripser_pd.size == 0:
		max_persistence = 0
	else:
		finite_bars_where = np.invert(np.isinf(ripser_pd[:,1]))
		finite_bars = ripser_pd[np.array(finite_bars_where),:]
		max_persistence = np.max(finite_bars[:,1]-finite_bars[:,0])

	return max_persistence


def getPHOutlierScores_multiDim(point_cloud,topological_radius,max_dim_ripser):

	set_of_super_outliers = np.empty((0,point_cloud.shape[1]))
	super_outlier_indices = np.empty((0,0), int)
	outlier_scores_point_cloud_original_order = np.empty((0,0))

	#for point in point cloud, get delta nhood
	for point_index in range(point_cloud.shape[0]):
		outlier_score = 0

		point = point_cloud[point_index,:]
		point_cloud_minus_point = np.delete(point_cloud,point_index,axis=0)

		kd_tree = KDTree(point_cloud_minus_point)
		indices = kd_tree.query_ball_point(point, r=topological_radius)

		number_of_neighbours = len(indices)


		if number_of_neighbours < 2:

			set_of_super_outliers = np.append(set_of_super_outliers, [point], axis=0)
			super_outlier_indices = np.append(super_outlier_indices,point_index)

		else:

			delta_point_cloud = point_cloud_minus_point[indices,:]

			diagrams = ripser(delta_point_cloud, maxdim=max_dim_ripser)['dgms']

			for dimension in range(max_dim_ripser+1):
				intervals = diagrams[dimension]
				max_persistence = getMaxPersistence(intervals)
				#print max_persistence
				if max_persistence > outlier_score:
					outlier_score = max_persistence


		outlier_scores_point_cloud_original_order = np.append(outlier_scores_point_cloud_original_order,outlier_score)

		#print("This is the outlier score", outlier_score)

	return outlier_scores_point_cloud_original_order,set_of_super_outliers, super_outlier_indices


def getPHOutlierScores_restrictedDim(point_cloud,topological_radius,dimension):

	set_of_super_outliers = np.empty((0,point_cloud.shape[1]))
	super_outlier_indices = np.empty((0,0), int)
	outlier_scores_point_cloud_original_order = np.empty((0,0))

	#for point in point cloud, get delta nhood
	for point_index in range(point_cloud.shape[0]):
		outlier_score = 0

		point = point_cloud[point_index,:]
		point_cloud_minus_point = np.delete(point_cloud,point_index,axis=0)

		kd_tree = KDTree(point_cloud_minus_point)
		indices = kd_tree.query_ball_point(point, r=topological_radius)

		number_of_neighbours = len(indices)


		if number_of_neighbours < 2:

			set_of_super_outliers = np.append(set_of_super_outliers, [point], axis=0)
			super_outlier_indices = np.append(super_outlier_indices,point_index)

		else:

			delta_point_cloud = point_cloud_minus_point[indices,:]

			diagrams = ripser(delta_point_cloud, maxdim=dimension)['dgms']

			intervals = diagrams[dimension]
			outlier_score = getMaxPersistence(intervals)


		outlier_scores_point_cloud_original_order = np.append(outlier_scores_point_cloud_original_order,outlier_score)

		#print("This is the outlier score", outlier_score)

	return outlier_scores_point_cloud_original_order, set_of_super_outliers, super_outlier_indices


#Landmark function

def getPHLandmarks(point_cloud, topological_radius, sampling_density, scoring_version, dimension, landmark_type):

	number_of_points = point_cloud.shape[0]
	number_of_PH_landmarks = int(round(number_of_points*sampling_density))

	if scoring_version == 'restrictedDim':

		outlier_scores_point_cloud_original_order,set_of_super_outliers, super_outlier_indices = getPHOutlierScores_restrictedDim(point_cloud,topological_radius,dimension)

	elif scoring_version == 'multiDim':

		max_dim_ripser = dimension
		outlier_scores_point_cloud_original_order,set_of_super_outliers, super_outlier_indices = getPHOutlierScores_multiDim(point_cloud,topological_radius,max_dim_ripser)


	number_of_super_outliers = set_of_super_outliers.shape[0]

	if landmark_type == 'representative':#small scores

		# sort outlier_scores_point_cloud_original_order
		sorted_indices_point_cloud_original_order = np.argsort(outlier_scores_point_cloud_original_order)

		#Permute zeros
		permuted_super_outlier_indices = np.random.permutation(sorted_indices_point_cloud_original_order[0:number_of_super_outliers])

		sorted_indices_point_cloud_original_order_without_super_outliers = np.array(sorted_indices_point_cloud_original_order[number_of_super_outliers:,])

		# We append the outliers at the end of the vector to select them last
		sorted_indices_point_cloud_original_order = np.append(sorted_indices_point_cloud_original_order_without_super_outliers,permuted_super_outlier_indices)

		PH_landmarks = point_cloud[sorted_indices_point_cloud_original_order[range(number_of_PH_landmarks)],:]

	elif landmark_type == 'vital':#large scores

		# sort outlier_scores_point_cloud_original_order
		sorted_indices_point_cloud_original_order = np.argsort(outlier_scores_point_cloud_original_order)

		#Permute zeros
		permuted_super_outlier_indices = np.random.permutation(sorted_indices_point_cloud_original_order[0:number_of_super_outliers])

		sorted_indices_point_cloud_original_order_without_super_outliers = np.array(sorted_indices_point_cloud_original_order[number_of_super_outliers:,])

		#We append the super outliers before the low scores so we select these last
		sorted_indices_point_cloud_original_order = np.append(permuted_super_outlier_indices,sorted_indices_point_cloud_original_order_without_super_outliers)

		# we flip the vector to keep in line with previous landmark call
		sorted_indices_point_cloud_original_order = np.flip(sorted_indices_point_cloud_original_order)

		PH_landmarks = point_cloud[sorted_indices_point_cloud_original_order[range(number_of_PH_landmarks)],:]


	return PH_landmarks, sorted_indices_point_cloud_original_order, number_of_super_outliers

# #####
# From here is our authorship

from scipy.spatial import Delaunay
from scipy.spatial import distance_matrix
from sklearn import datasets
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import tensorflow as tf



def Sort_Tuple(tup):
  """
  Input:  tup= a list of tuples (j,xij) where j is the index of the point uj in U
  and xij is the barycentric coordinate associated to uj of a point x in R^n
  Output: the list tup sorted by the influences of the points of U on the point x
  (i.e., sorted by xij). This function will be used to explain the result obtained
  by the SMNN
  """
  lst = len(tup)
  for i in range(lst):

        for j in range(lst-i-1):
            if (tup[j][1] > tup[j + 1][1]):
                temp = tup[j]
                tup[j] = tup[j + 1]
                tup[j + 1] = temp

  return tup

def representative_Us(V,epsilon=0.1):
    """
    Input:  V=set of points in R^n,
            epsilon=real number>0
    Output: V[dom] =  Subset of V with  representativeness factor epsilon
                      (i.e, each point of V is at distance less or equal to epsilon
                      to a point of V[dom])
    """
    ady = distance_matrix(V,V)
    g = nx.from_numpy_matrix(ady<epsilon)
    dom = np.array(list(nx.dominating_set(g)))

    return V[dom]

def hyperplane(ps):
    """
    Input: ps=set of n-1 points in R^n
    Output: coeffs= coefficients and c=independent term of the equation of the
    hyperplane determined by the points ps
    """
    b=np.array([1]*len(ps))
    coeffs = np.linalg.solve(ps,b)
    c=-np.sum(np.multiply(ps[0],coeffs))

    return coeffs, c

center_function = lambda X: X - X.mean() #Move the set X so that its center of mass is the coordinate origin

def boundary(tri):
    """
    Input:  tri=a Delaunay triangulation
    Output: the boundary of tri. That is,
            np.stack(n_simplices)=a set of n-simplices and
            np.stack(n_1_simplices)=a set of (n-1)-simplices
            such that each (n-1)-simplex is shared by exactly one n-simplex of tri.
    """
    maximal_simplices = tri.simplices
    neighbors = tri.neighbors
    n_1_simplices = list()
    n_simplices = list()
    for i in range(len(maximal_simplices)):
        neighbor = neighbors[i]
        if (-1) in neighbor:
            simplex = maximal_simplices[i]
            for j in range(len(simplex)):
              if neighbor[j]==-1:
                face = np.delete(simplex,j)
                n_1_simplices.append(face)
                n_simplices.append(simplex)

    return np.stack(n_simplices), np.stack(n_1_simplices)

def barycentric_respect_to_del(vs,tri,R=10,maximal_simplexes=[], boundary_tri=[]):
  """
  Input:  vs= set of points in R^n
          tri=a Delaunay triangulation in R^n
          R= radio of the hypersphere
          maximal_simplices=maximal simplices of tri (if we have precomputed it, we don't compute it again)
          boundary_tri=the output of boundary(tri) (if we have precomputed it, we don't compute it again)
  Output: np.stack(bss)=the set of xi(v) for each point v of vs,
          where xi_j(v) corresponds to the barycentric coordinates b(v) of v associated to vertex uj in the triangulation tri
          maximal_simplices=maximal simplices of tri
          boundary_tri=the output of boundary(tri)
  """
  n=len(vs)
  u_len = len(tri.points)
  l_simplex = len(tri.simplices[0])
  bss = list()
  if len(maximal_simplexes)==0:
      maximal_simplexes, boundary_tri=boundary(tri)
      print("Boundary already computed")
  counter=0
  for v in vs:
    print(counter," of ",n)
    counter+=1
    i= tri.find_simplex(v)
    if i!=-1:
        simplex = tri.simplices[i]
        b=tri.transform[i,:len(simplex)-1].dot(np.transpose([v] - tri.transform[i,len(simplex)-1]))
        b= np.c_[np.transpose(b), 1 - b.sum(axis=0)][0]
        u = np.array([0.0]*u_len)
        for j in range(len(simplex)):
            u[simplex[j]]=b[j]
        bss.append(u)
    else:
        print("Found a point v outside the triangulation")
        P=R*v/np.linalg.norm(v)
        for j in range(len(boundary_tri)):
            sigma = boundary_tri[j] #Ex. sigma=[2, 54,13]
            max_simplex = maximal_simplexes[j]
            p0=np.setdiff1d(max_simplex,sigma) #p0=vertex not in sigma but in the maximal simplex containing sigma
            p0=tri.points[p0][0] #coordinates
            points = tri.points[sigma] #coordinates
            coeffs_hyp, c = hyperplane(points)
            subst_p0_hyp=np.sum(np.multiply(p0,coeffs_hyp))+c
            subst_v_hyp=np.sum(np.multiply(v,coeffs_hyp))+c
            if (subst_p0_hyp * subst_v_hyp) < 0:
                ps = np.concatenate((tri.points[sigma],[P]))
                tri2 = Delaunay(ps)
                b=tri2.transform[0,:l_simplex-1].dot(np.transpose([v] - tri2.transform[0,l_simplex-1]))
                b= np.c_[np.transpose(b), 1 - b.sum(axis=0)][0]
                cond1 = all(b>=0)
                if cond1:
                    break
        print("Found a face mu in the boundary of tri")
        u = np.array([0.0]*u_len)
        for j in range(len(sigma)):
            u[sigma[j]]=b[j]  ##Ex. sigma=[2, 54,13], sigma[0]=2,  u[2]=b[0]
        bss.append(u)
        print("Barycentric coordinates  of v computed.")

  return np.stack(bss), maximal_simplexes, boundary_tri


def SMNN(V,y,epochs,epsilon_param=10,method=0,sd = 0.6,dim = 0,maxdim=1,verbose=False):

    """
    Train a SMNN on a dataset V with y labels during a certain number of epochs
    and computes a representative dataset with epsilon equals to R/epsilon_param.

     Input:  V= set of points in R^n
             y= set of labels where i-th label in y corresponds to i-th vertex in V
             epoch=number of epochs to train the SMNN
             epsilon_param =>  used to compute a dataset Us
             method= 0 for representative datasets, 
                     1 for representative PH landmarks (restricted dim), 
                     2 for vital PH landmarks (restricted dim),
                     3 for representative PH landmarks (multi dim),
                     4 for vital PH landmarks (multi dim)

     Output:   bis= the set of points xi(v) for each point v of V
               y_hot= one_hot vectors associated to labels y
               Us= representative dataset with representativeness factor epsilon = R/epsilon_param
               tri=Delaunay triangulation with vertex set Us
               model= tf.keras.Sequential()
               history=model.fit(...)
               m=Us.mean()
               R= 0.5 + the maximum distance from V.mean() to any point of V used to compute epsilon
               maximal_simplices=maximal simplices of tri
               boundary_tri=the output of boundary(tri)
    """
    n_features = np.shape(V)[1]
    R=np.max(distance_matrix([[0]*n_features],center_function(V))[0])+0.5
    eps=R/epsilon_param
    # -------
    if method==0:
        Us = representative_Us(V,eps)
        print("Using representative dataset of size",len(Us))
    # -------
    tp = R/2
    print("Radius: ", R, ", and topological radius: ",tp)
    
    # -------
    if method==1:
        Us=getPHLandmarks(V, topological_radius=tp, sampling_density=sd, scoring_version='restrictedDim', dimension=dim, landmark_type='representative' )[0]
        print("Using PH representative landmark (restricted dim) of size",len(Us))
    # -------
    if method==2:
        Us=getPHLandmarks(V, topological_radius=tp, sampling_density=sd, scoring_version='restrictedDim', dimension=dim, landmark_type='vital' )[0]
        print("Using PH vital landmark (restricted dim) of size",len(Us))
    # -------
    if method==3:
        Us=getPHLandmarks(V, topological_radius=tp, sampling_density=sd, scoring_version='multiDim', dimension=maxdim, landmark_type='representative' )[0]
        print("Using PH representative landmark (multi dim) of size",len(Us))
    # -------
    if method==4:
        Us=getPHLandmarks(V, topological_radius=tp, sampling_density=sd, scoring_version='multiDim', dimension=maxdim, landmark_type='vital' )[0]
        print("Using PH vital landmark (multi dim) of size",len(Us))
        
    m=Us.mean()
    print("Centering Us")
    Us=Us-m
    print("Centering V with respect to the center of mass of Us")
    V=V-m

    #%%
    tri = Delaunay(Us)
    print("Delaunay triangulation already computed")

    #%% labels update
    n_classes = len(set(y))
    y_hot=tf.one_hot(y,depth=n_classes)
    y_hot=np.array(y_hot)

    #%%
    bis, maximal_simplexes, boundary_tri=np.array(barycentric_respect_to_del(V,tri,R))
    print("Barycentric coordinates already computed.")
    input_dim = np.shape(bis)[1]

    #%%
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(n_classes, activation="softmax",use_bias=False, input_shape=(input_dim,)))
    loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    model.compile(optimizer="sgd",loss="categorical_crossentropy",metrics=["accuracy"])
    print("Training neural network...")
    history = model.fit(bis,y_hot,epochs=epochs, verbose = verbose)

    return (bis, y_hot, Us, tri, model, history,m, R,maximal_simplexes, boundary_tri)
