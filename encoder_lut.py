import torch
import torch.nn.functional as F
import math
from tqdm import tqdm
from fuzzy_feature import get_fuzzy_features3
from sklearn.metrics.pairwise import cosine_similarity


cos_bins = [-1.0000e+00, -8.3333e-01, -6.6667e-01, -5.0000e-01, -3.3333e-01,\
        -1.6667e-01, -2.9802e-08,  1.6667e-01,  3.3333e-01,  5.0000e-01,\
         6.6667e-01,  8.3333e-01,  1.0000e+00] # cos_bins = torch.linspace(-1, 1, 13)

#triangle_val = torch.tensor([[ 1.7321e-01,  1.0000e-01,  1.2246e-17, -1.0000e-01, -1.7321e-01,
#         -2.0000e-01, -1.7321e-01, -1.0000e-01, -3.6739e-17,  1.0000e-01,
#          1.7321e-01,  2.0000e-01],
#        [ 1.0000e-01,  1.7321e-01,  2.0000e-01,  1.7321e-01,  1.0000e-01,
#          2.4493e-17, -1.0000e-01, -1.7321e-01, -2.0000e-01, -1.7321e-01,
#         -1.0000e-01, -4.8986e-17]]) #diff=4/node_size


def make_rads(node_size):
  angle_list = [30,60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
  cos_list=[]
  sin_list=[]
  diff = 4/node_size  #반경 4/node_size으로 설정
  #diff = 8/node_size  #반경 4/node_size으로 설정
  #diff = 16/node_size
  
  for i in range(len(angle_list)):
    rad = math.radians(angle_list[i])
    cos_list.append(diff * math.cos(rad))
    sin_list.append(diff * math.sin(rad))
  cos_value = torch.tensor(cos_list)
  sin_value = torch.tensor(sin_list)

  triangle_value = torch.stack([cos_value, sin_value], dim=1)
  tri=triangle_value.transpose(1,0)
  # tri = [2, 12] 
  # first row = r * cos_theta(rad)
  # second row = r * sin_theta(rad)


  del triangle_value, cos_value, sin_value, cos_list, sin_list

  return tri

def get_large(a,b):
  if a >= b :
    large = a
    small = b
  else :
    large = b
    small = a 

  return large,small

def count_nodes(base1, base2, graph, diff): # given node와 knot 1개가 주어졌을 때 해당하는 정점의 수
  p1_x = base1[0]
  p3_x = base1[0]
  p2_x = base2[0]
  p4_x = base2[0]
  
  p1_y = base1[1]+ diff
  p3_y = base1[1]- diff
  p2_y = base2[1]+ diff
  p4_y = base2[1]- diff


  #print(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y)
  large_x, small_x = get_large(p1_x, p2_x)

  
  large_y = max(p1_y, p2_y, p3_y, p4_y)
  small_y = min(p1_y, p2_y, p3_y, p4_y)

  #coordinate x
  table_x_con1 = graph[:,0] <= large_x
  table_x_con2 = graph[:,0] >= small_x
  table_x = table_x_con1 * table_x_con2 
  
  #coordinate y
  table_y_con1 = graph[:,1] <= large_y
  table_y_con2 = graph[:,1] >= small_y
  table_y = table_y_con1 * table_y_con2 

  table=table_x *  table_y
  #heatmap.sum(dim = -1)

  
  return table.sum(dim=-1)

def make_knots(given_nodes, tri_value, num_of_directions = 12):
  d1, d2 = given_nodes.size() 

  #그래프 복사
  knots = torch.zeros([d1, d2] ,dtype=given_nodes.dtype)
  knots[:,:]=given_nodes[:,:] 
  knots = knots.unsqueeze(-1).repeat(1,1,12)

  #x1 + cos theta, x2 + sin theta
  knots = knots[:,:,:] + tri_value

  return knots

def get_count_matrix(dist_matrix, dist_max, dist_min=None):
    if dist_min!=None:
      heatmap1 = dist_matrix >= dist_min
      heatmap2 = dist_matrix < dist_max
      heatmap = heatmap1 * heatmap2
    else:
      heatmap = dist_matrix <=dist_max
      
    count_matrix = heatmap.sum(dim = -1)
    
    return count_matrix

def get_one_hot_encoding(count_nodes, num_categories):
    # 1~num_cate 의 category에 할당됨. 
    bins = torch.linspace(1, count_nodes.max() + 1, num_categories)#.to("cuda")
    #print("bins ", bins)
    category_indices = torch.bucketize(count_nodes, bins, right = False)
    #print("cate ", category_indices)
    one_hot_categories = torch.nn.functional.one_hot(category_indices, num_categories)
    #print(one_hot_categories)
    return one_hot_categories

def get_embedding2(dist_matrix, node_size, bucket_size):
    
    range1 = 4/node_size
    range2 = 8/node_size


    count_matrix1 = get_count_matrix(dist_matrix,  range1) # range1 구간의 개수 counting, [N]
    count_matrix2 = get_count_matrix(dist_matrix,  range2) # range2 구간의 개수 counting, [N]

    #print("count_matrix1 ", count_matrix1)
    #print("count_matrix2 ", count_matrix2)
    
    one_hot1 = get_one_hot_encoding(count_matrix1,  bucket_size)
    one_hot2 = get_one_hot_encoding(count_matrix2,  bucket_size)
    
    embedding2 = torch.concat([one_hot1, one_hot2], dim = -1)
    return embedding2

def get_embedding5(tsp_instance, knots, node_size, diff, bucket_size=4):
    tmp = []
    c_mat = []
    
    for j in range(12):
        for i in range(node_size):
            tmp.append(count_nodes(tsp_instance[i], knots[i, :, j], tsp_instance, diff=diff))
        c_mat.append(torch.tensor(tmp))
       
        tmp = []
    
    

    one_hot1 = get_one_hot_encoding(c_mat[0], bucket_size)
    one_hot2 = get_one_hot_encoding(c_mat[1], bucket_size)
    one_hot3 = get_one_hot_encoding(c_mat[2], bucket_size)
    one_hot4 = get_one_hot_encoding(c_mat[3], bucket_size)
    one_hot5 = get_one_hot_encoding(c_mat[4], bucket_size)
    one_hot6 = get_one_hot_encoding(c_mat[5], bucket_size)
    one_hot7 = get_one_hot_encoding(c_mat[6], bucket_size)
    one_hot8 = get_one_hot_encoding(c_mat[7], bucket_size)
    one_hot9 = get_one_hot_encoding(c_mat[8], bucket_size)
    one_hot10 = get_one_hot_encoding(c_mat[9], bucket_size)
    one_hot11 = get_one_hot_encoding(c_mat[10], bucket_size)
    one_hot12 = get_one_hot_encoding(c_mat[11], bucket_size)

    embedding5 = torch.concat([one_hot1, one_hot2, one_hot3, one_hot4, one_hot5, one_hot6,
                               one_hot7, one_hot8, one_hot9, one_hot10, one_hot11, one_hot12], dim=-1)
    
    return embedding5

def get_angle_matrix(node_positions):
    delta_xs = node_positions[:, 0].unsqueeze(dim=1) - node_positions[:, 0].unsqueeze(dim=-1)
    delta_ys = node_positions[:, 1].unsqueeze(dim=1) - node_positions[:, 1].unsqueeze(dim=-1)
    angle_matrix = torch.atan2(delta_ys, delta_xs)
    return angle_matrix

def get_embedding3(dist_matrix, angle_matrix, bucket_size):
    num_sectors = 12
    sector_angle = 2 * torch.pi / num_sectors

    one_hots = []
    max_distance = dist_matrix.max()

    for i in range(num_sectors):
        sector_start = -torch.pi + i * sector_angle
        sector_end = -torch.pi + (i + 1) * sector_angle
        in_sector_mask = (angle_matrix > sector_start) & (angle_matrix <= sector_end)

        count_matrix = get_count_matrix(dist_matrix * in_sector_mask, max_distance / 8)
        one_hot = get_one_hot_encoding(count_matrix, bucket_size)
        one_hots.append(one_hot)

    embedding3 = torch.concat(one_hots, dim=-1)
    return embedding3

def get_embedding4(region, dist_matrix, node_size,  bucket_size):

    range1 = 4/node_size
   
    region_mask = region <= range1
    
    mask = torch.empty(region_mask.size())
    mask[region_mask==False] = -10
    mask[region_mask==True] = 1
    
  
    dist_matrix = dist_matrix * mask

    count_matrix2 = get_count_matrix(dist_matrix,  cos_bins[1], cos_bins[0])
    count_matrix3 = get_count_matrix(dist_matrix,  cos_bins[2], cos_bins[1])
    count_matrix4 = get_count_matrix(dist_matrix,  cos_bins[3], cos_bins[2])
    count_matrix5 = get_count_matrix(dist_matrix,  cos_bins[4], cos_bins[3])
    count_matrix6 = get_count_matrix(dist_matrix,  cos_bins[5], cos_bins[4])
    count_matrix7 = get_count_matrix(dist_matrix,  cos_bins[6], cos_bins[5])

    
    count_matrix8 = get_count_matrix(dist_matrix,  cos_bins[7], cos_bins[6])
    count_matrix9 = get_count_matrix(dist_matrix,  cos_bins[8], cos_bins[7])
    count_matrix10 = get_count_matrix(dist_matrix,  cos_bins[9], cos_bins[8])
    count_matrix11 = get_count_matrix(dist_matrix,  cos_bins[10], cos_bins[9])
    count_matrix12 = get_count_matrix(dist_matrix,  cos_bins[11], cos_bins[10])
    count_matrix13 = get_count_matrix(dist_matrix,  cos_bins[12], cos_bins[11])
    

    
    one_hot2 = get_one_hot_encoding(count_matrix2,  bucket_size)
    one_hot3 = get_one_hot_encoding(count_matrix3,  bucket_size)
    one_hot4 = get_one_hot_encoding(count_matrix4,  bucket_size)
    one_hot5 = get_one_hot_encoding(count_matrix5,  bucket_size)
    one_hot6 = get_one_hot_encoding(count_matrix6,  bucket_size)
    one_hot7 = get_one_hot_encoding(count_matrix7,  bucket_size)
    
    one_hot8 = get_one_hot_encoding(count_matrix8,  bucket_size)
    one_hot9 = get_one_hot_encoding(count_matrix9,  bucket_size)
    one_hot10 = get_one_hot_encoding(count_matrix10,  bucket_size)
    one_hot11 = get_one_hot_encoding(count_matrix11,  bucket_size)
    one_hot12 = get_one_hot_encoding(count_matrix12,  bucket_size)
    one_hot13 = get_one_hot_encoding(count_matrix13,  bucket_size)
    

    
    
    embedding3 = torch.concat([one_hot2, one_hot3, one_hot4, one_hot5, one_hot6,\
     one_hot7, one_hot8, one_hot9, one_hot10, one_hot11, one_hot12, one_hot13], dim = -1)
    
    return embedding3


def get_encoder_embedding(tsp_instance, node_size, depth=6, diff= 1/(20*1) , bucket_size=4):
    
    dist_matrix = torch.cdist(tsp_instance, tsp_instance)
    #dist_matrix = torch.cdist(tsp_instance.to("cuda"), tsp_instance.to("cuda"))


    embedding1 = get_fuzzy_features3(tsp_instance, depth)
    embedding2 = get_embedding2(dist_matrix, node_size, bucket_size)
  
    cos_mat=cosine_similarity( tsp_instance-0.5, tsp_instance-0.5)
    cos_mat = torch.tensor(cos_mat)
    embedding3 = get_embedding4(dist_matrix, cos_mat, node_size,  bucket_size)


    #angle_matrix = get_angle_matrix(tsp_instance)
    #embedding3 = get_embedding3(dist_matrix, angle_matrix, bucket_size)
    
    
    #triangle_val = make_rads(node_size=node_size)
    #knots = make_knots(tsp_instance-0.5, triangle_val)
    #embedding3 = get_embedding5(tsp_instance-0.5, knots, node_size, diff,  bucket_size)
 

   
    embedding = torch.concat([embedding1, embedding2,  embedding3], dim = -1)
    #embedding = torch.concat([embedding1, embedding2], dim = -1)
    #dembedding = embedding1

    """
    #dist 반영
    dist_matrix = dist_matrix.to(dtype=embedding.dtype)
    dist_matrix = dist_matrix * (-1)
  
    #print("\n dist ", dist_matrix.size(), embedding.size())
    embedding = torch.matmul(dist_matrix, embedding)
    """
    
    return embedding

