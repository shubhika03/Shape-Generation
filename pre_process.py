import open3d as o3d
import numpy as np


class Node: 
	def __init__(self,key): 
		self.left = None
		self.right = None
		self.val = key


def make_kd_tree(points, dim=3, i=0):

	#kd tree algo
	if points.shape[0] > 1:
		points = points[points[:, i].argsort()]

		i = (i + 1) % dim
		half = len(points) >> 1

		root = Node(points[half])
		root.left = make_kd_tree(points[: half], dim, i)
		root.right = make_kd_tree(points[half + 1:], dim, i) 
		return root
		
	elif points.shape[0] == 1:
		return Node(points[0])

def printInorder(root, sortedorder): 

	#inorder traversal on kd tree
  
	if root: 
		printInorder(root.left, sortedorder) 
		sortedorder.append(root.val)
		printInorder(root.right, sortedorder) 

def swap(mat, col, row1, row2):

	#function to swap 2 rows given the column of a matrix

	temp = mat[row1, col]
	mat[row1, col] = mat[row2, col]
	mat[row2, col] = temp

def change_ind(all_pts_vector, shape_no, first_ind, second_ind):

	#helper function to help in swapping in iterative point ordering

	all_pts_vector_temp = all_pts_vector.copy()
	swap(all_pts_vector_temp, shape_no, first_ind, second_ind)
	swap(all_pts_vector_temp, shape_no, first_ind+1000, second_ind+1000)
	swap(all_pts_vector_temp, shape_no, first_ind+2000, second_ind+2000)
	return all_pts_vector_temp

def change_ind_diff(all_pts_vector, shape_no, first_ind, second_ind):

	#helper function to help in swapping in iterative point ordering

	temp= np.zeros((6, all_pts_vector.shape[1]))
	temp[0] = all_pts_vector[first_ind, :]
	temp[1] = all_pts_vector[second_ind, :]
	temp[2] = all_pts_vector[first_ind+1000, :]
	temp[3] = all_pts_vector[second_ind+1000, :]
	temp[4] = all_pts_vector[first_ind+2000, :]
	temp[5] = all_pts_vector[second_ind+2000, :]

	swap(temp, shape_no, 0, 1)
	swap(temp, shape_no, 2, 3)
	swap(temp, shape_no, 4, 5)
	

	return temp

def pca(centered_pts):
	#function to calculate egien vectors and eigen values and intital loss due to dimension reduction
	var = np.cov(centered_pts.T)
	values, vectors = np.linalg.eig(var)

	recon = centered_pts@vectors[:, :100]@vectors[:, :100].T + mu
	init_loss = np.sum((recon - all_pts_vector)**2)

	return values, vectors, init_loss


#vector to store all initial ordered points
all_pts_vector = []

for i in range(5000):
	#read pcd point clouds
	pcd = o3d.io.read_point_cloud('shapenet-chairs-pcd/'+ str(i+1) + '.pcd')
	out_arr = np.asarray(pcd.points)

	#kd tree algo
	root = make_kd_tree(out_arr)

	sortedorder = []
	printInorder(root, sortedorder)
	sortedorder = np.array(sortedorder)

	#store flatten points
	flatten_list = []
	for j in range(3):
		flatten_list.extend(sortedorder[:, j])

	all_pts_vector.append(flatten_list)


#3N X S = 3000 X 5000 vector 
all_pts_vector = np.array(all_pts_vector).T

#subtract mean
mu = np.mean(all_pts_vector, axis=0)
centered_pts = all_pts_vector - mu

#pca to calcualte basis
values, vectors, init_loss = pca(centered_pts)

basis = centered_pts@vectors[:, :100]

#temp vectors to make calucation in iterative point ordering easier
product_vector = vectors[:, :100]@vectors[:, :100].T
init_recon = centered_pts@product_vector

#initial loss due to pca
print(init_loss)

loss_at_iter = []
loss_at_iter.append(init_loss)


#iterative point ordering
for i in range(1000):
	print("***************** " + str(i))
	for j in range(5000):  # for each shape
		print("&&&&&&&&&& " + str(j))
		for k in range(10000):
			first_ind = np.random.randint(0, 1000)
			second_ind = np.random.randint(0, 1000)

			#swap indices and calculate loss

			temp_change_mat = change_ind_diff(centered_pts, j, first_ind, second_ind)
			temp_product_mat = temp_change_mat@product_vector
			swap_recon = init_recon.copy()
			swap_recon[first_ind, :] = temp_product_mat[0]
			swap_recon[second_ind, :] = temp_product_mat[1]
			swap_recon[first_ind+1000, :] = temp_product_mat[2]
			swap_recon[second_ind+1000, :] = temp_product_mat[3]
			swap_recon[first_ind+2000, :] = temp_product_mat[4]
			swap_recon[second_ind+2000, :] = temp_product_mat[5]

			centered_pts_temp = change_ind(centered_pts, j, first_ind, second_ind)

			swap_loss = np.sum((swap_recon - centered_pts_temp)**2)
			
			if swap_loss < init_loss:
				print(swap_loss)
				print(init_loss)
				centered_pts = centered_pts_temp
				init_loss = swap_loss
				init_recon = centered_pts@product_vector

	#calculate new asis after every epoch
	values, vectors, init_loss = pca(centered_pts)
	product_vector = vectors[:, :100]@vectors[:, :100].T
	init_recon = centered_pts@product_vector
	loss_at_iter.append(init_loss)
	print(loss_at_iter)

print(loss_at_iter)
np.save("mu.npy", mu)
np.save("basis.npy", basis)
np.save("coefficient.npy", vectors[:, :100])
