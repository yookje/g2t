import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor

#band_debth_matrix=[1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10]

mid_points_matrix=[[[-1/2**(d+2),-1/2**(d+2)],[1/2**(d+2),-1/2**(d+2)],[-1/2**(d+2),1/2**(d+2)],[1/2**(d+2),1/2**(d+2)]]for d in range(9)]
band_debth_matrix=[1/18, 1/15, 1/12, 1/9, 1/6, 1/3,1/3 ,1/3, 1/3]
grid_size_matrix = [1,1/2,1/4,1/8,1/16,1/32,1/64,1/128,1/256]

box_to_binary = {
    2: [0, 0, 0, 1],
    3: [0, 1, 0, 0],
    1: [0, 0, 1, 0],
    4: [1, 0, 0, 0]
}
half_band_matrix= [band_debth_matrix[ii]*grid_size_matrix[ii] for ii in range(len(grid_size_matrix))]




def get_band_ratio(depth):
    
    if depth>9:
        print("depth should not be over 9")
        return
    return band_debth_matrix[depth-1]


#Huh Joon code start

def process_point(point_idx, coordinates, mid_points):
    # 각 점에 대해 모든 depth를 처리하고 비트 결과를 연결
    point = coordinates[point_idx]
    bit_result = []
    
    #print(mid_points)
    for depth in range(len(mid_points[point_idx])):
        mid_point = mid_points[point_idx][depth]
        half_band_width = half_band_matrix[depth]
        bits = get_four_bits(point, mid_point, half_band_width)
        #print(bits)
        bit_result.extend(bits)
        #print(bit_result)
    return bit_result


def get_four_bits(point, mid_point, half_band_width):
    mid_x, mid_y = mid_point
    up = point[1] > (mid_y - half_band_width)
    down = point[1] < (mid_y + half_band_width)
    right = point[0] > (mid_x - half_band_width)
    left = point[0] < (mid_x + half_band_width)
    
    #print(point)
    #print(mid_point)
    #print(up,down,right,left)
    
    temp = [left & right & up, right & down & up, left & right & down, left & up & down]
    #print(temp)
    return [int(element) for element in temp]


def apply_shifts_for_column(initial_center, shifts_column):
    center = initial_center
    centers_for_column = []
    for shift in shifts_column:
        center = apply_single_shift(center, shift)
        centers_for_column.append(center)
    return centers_for_column
def apply_single_shift(center, shift):
    return (center[0] + shift[0], center[1] + shift[1])



def get_fuzzy_features3(coordinates,depth):

    coordinates=np.array(coordinates)
    
    #coordinates: n*2짜리
    new_coordinates=coordinates*(2**(depth+1))
    xs = new_coordinates[:,0]
    ys = new_coordinates[:,1]
    
    
    rights= np.array([(xs%(2**(value+1))//(2**value)).astype(int) for value in range(depth,0,-1)])
    ups=np.array([(ys%(2**(value+1))//(2**value)).astype(int) for value in range(depth,0,-1)])
    
    boxes = 1+rights+2*ups
    
    boxes_transposed = boxes.T

    # Define the mapping from box numbers to their respective binary representations


    # Convert the boxes values to their binary representations
    binary_representation = np.array([[box_to_binary[box] for box in row] for row in boxes_transposed])
    extended_binary_representation = np.array([np.concatenate(row) for row in binary_representation])

    
    grid_coord_shifts = [[mid_points_matrix[d][b-1] for b in row] for d,row in enumerate(boxes)]
    
    #print(grid_coord_shifts)
    
    initial_center=[0.5,0.5]
    
    final_results = []

    for column_index in range(len(grid_coord_shifts[0])):  # Assuming all columns have the same number of shifts
        # Apply the first shift in the column
        shifts_column = [row[column_index] for row in grid_coord_shifts if column_index < len(row)]
        # Apply the shifts for the current column
        column_results = apply_shifts_for_column(initial_center, shifts_column)
        bef=[(0.5,0.5)]
        bef.extend(column_results)
        bef=bef[:-1]
        final_results.append(bef)
        

    n = len(coordinates)
    results = [None] * n
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_point, i, coordinates, final_results) for i in range(n)]
        for future in futures:
            idx = futures.index(future)
            results[idx] = future.result()
    
    #print(extended_binary_representation)
    #print(results)
    
    
    fin_result = []

    # A와 B의 각 행을 반복하면서 처리합니다.
    for row_a, row_b in zip(extended_binary_representation, results):
        # 새로운 행을 생성하기 위한 임시 리스트를 초기화합니다.
        new_row = []

        # 각 행의 길이를 4개씩 나누어 번갈아 가면서 합치되, 마지막 4개를 제외하고 번갈아가며 합칩니다.
        for i in range(0, len(row_a),4):
            new_row.extend(row_a[i:i+4])  # A 배열에서 4개의 요소를 추출하여 새로운 행에 추가합니다.
        # 마지막에 B 배열의 전체를 추가합니다.
            new_row.extend(row_b[i:i+4])

        # 생성된 새로운 행을 결과 배열에 추가합니다.
        fin_result.append(new_row)

            
            
    return torch.tensor(fin_result)
    
#huhjooncode end


import time
if __name__ == "__main__":
    
    start_time = time.time()
    node_size = 2
    #coordinates = torch.rand(node_size, 2)
    #coordinates = points = torch.rand(20000, 2)
    coordinates=torch.tensor([[0.9062, 0.1269],[0.9501, 0.5458]])
    #print(coordinates)
    one_hot_encoded = get_fuzzy_features3(coordinates, depth =6 )
    print("===============")
    print(one_hot_encoded)
    



    print(one_hot_encoded.size())
    #print(one_hot_encoded.shape)
    
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time

    print(elapsed_time)
 