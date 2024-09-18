# Copyright (c) 2020, Jefferson Science Associates, LLC. All Rights Reserved. Redistribution
# and use in source and binary forms, with or without modification, are permitted as a
# licensed user provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this
#    list of conditions and the following disclaimer in the documentation and/or other
#    materials provided with the distribution.
# 3. The name of the author may not be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# This material resulted from work developed under a United States Government Contract.
# The Government retains a paid-up, nonexclusive, irrevocable worldwide license in such
# copyrighted data to reproduce, distribute copies to the public, prepare derivative works,
# perform publicly and display publicly and to permit others to do so.
#
# THIS SOFTWARE IS PROVIDED BY JEFFERSON SCIENCE ASSOCIATES LLC "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL
# JEFFERSON SCIENCE ASSOCIATES, LLC OR THE U.S. GOVERNMENT BE LIABLE TO LICENSEE OR ANY
# THIRD PARTES FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import tensorflow as tf
import numpy as np

def find_pareto_front(solutions):
    """
    """
    if solutions.shape[0] == 0:
        return solutions
    
    # Sort on first dimension (descending value) (weighted sum method using the sum of all elements)
    tempArray = -1*np.array([sum(x) for x in myArray])
    myArray[:] = myArray[tempArray.argsort()][::-1]
    pareto_frontier = None
    # Test next rows against the last row in pareto_frontier
    flag = True
    while True:
        # The top element is by definition Pareto (or it would have been removed by domination)
        if flag:
            pareto_frontier = myArray[0:1,:]
            flag = False
        else:
            pareto_frontier = np.concatenate((pareto_frontier, myArray[0:1,:]))

        # remove the Pareto point that we've added to the Pareto frontier
        myArray = np.delete(myArray, 0, 0)

        rowNr = 0
        while len(myArray) != 0 and rowNr < len(myArray):
            row = myArray[rowNr:rowNr+1,:][0]
            if sum([row[x] >= pareto_frontier[-1][x]
                    for x in range(len(row))]) == len(row):
                # If it is worse on all features remove the row from the array
                myArray = np.delete(myArray, rowNr, 0)
            else:
                rowNr += 1

        if len(myArray) == 0:
            break
    return pareto_frontier


def vectorized_cummin(tensor):
    # Define a function that computes the minimum of two values
    def min_fn(prev_min, current):
        return tf.minimum(prev_min, current)

    # Use tf.scan to compute the cumulative minimum
    cum_min = tf.scan(min_fn, tensor, initializer=tensor[0])  # Start with the first element as the initial minimum
    return cum_min
    
def find_pareto_front_tf(x, y):
    # Step 1: Find the minimum of x
    xmin = tf.reduce_min(x)
    
    # Step 2: Create a filter for the minimum x
    xfilter = tf.equal(x, xmin)
    
    # Step 3: Find the minimum of y where x is xmin
    ymin = tf.reduce_min(tf.boolean_mask(y, xfilter))
    
    # Step 4: Create a filter for y values less than ymin
    yfilter = tf.less(y, ymin)
    
    if tf.reduce_any(yfilter):
        # Step 5: Filter x and y based on yfilter
        x_filtered = tf.boolean_mask(x, yfilter)
        y_filtered = tf.boolean_mask(y, yfilter)
        
        # Step 6: Sort filtered x and get indices
        sorted_indices = tf.argsort(x_filtered)
        x_sorted = tf.gather(x_filtered, sorted_indices)
        y_sorted = tf.gather(y_filtered, sorted_indices)

        # Step 7: Cumulative minimum
        # cum_min = tf.math.cummin(y_sorted)
        cum_min = vectorized_cummin(y_sorted)

        # Step 8: Create final filter
        yfilter_final = tf.concat([[True], tf.less(y_sorted[1:], cum_min[:-1])], axis=0)

        # Step 9: Create result
        result = tf.stack([tf.concat([[xmin], tf.boolean_mask(x_sorted, yfilter_final)], axis=0),
                           tf.concat([[ymin], tf.boolean_mask(y_sorted, yfilter_final)], axis=0)], axis=1)
        
        return result  # Convert to NumPy array if needed
    else:
        return tf.stack([[xmin], [ymin]], axis=1)
    
    
# def hypervolume(solutions, ref):
#     """
#     """
#     if solutions.shape[0] == 0:
#         return tf.convert_to_tensor([0.0])
    
#     ref = tf.convert_to_tensor(ref)
#     good_indices = np.where((solutions[:, 0] <= ref[0])&(solutions[:, 1]<= ref[1]))[0]
#     solutions = tf.gather(solutions, good_indices)
    
#     # solutions = find_pareto_front_tf(solutions[:, 0], solutions[:, 1])
    
#     indices = tf.argsort(solutions[:, 0], axis=-1, direction='DESCENDING', stable=False, name=None)
#     x_reordered = tf.gather(solutions[:, 0], indices)
#     y_reordered = tf.gather(solutions[:, 1], indices)

#     edge_y = 0
#     new_edge_y = 0
#     tf_perato_area = 0
#     total_points = 0
#     for i in range(0,len(x_reordered)-1):
    
#         delta_x = tf.where(x_reordered[i + 1] < ref[0], x_reordered[i + 1] - ref[0], 0) # tf.abs(x_reordered[i + 1] - ref[0])
#         delta_y = tf.where(y_reordered[i] < ref[1], y_reordered[i + 1] - y_reordered[i], 0) #tf.abs(y_reordered[i + 1] - y_reordered[i])
            
#         new_edge_y = tf.where(y_reordered[i + 1] > edge_y, y_reordered[i + 1], edge_y)
#         delta_y = tf.where(y_reordered[i + 1] > edge_y, tf.abs(y_reordered[i + 1] - edge_y), 0)
        
#         area = delta_x.numpy()*delta_y.numpy()
#         tf_perato_area += area
#         total_points += tf.where(delta_y>0, 1, 0)
#         edge_y = new_edge_y
        
#     # Add last bit to match PyMOO #################### Blame Kishan ##############
#     delta_x = tf.where(x_reordered[len(x_reordered)-1] < ref[0], x_reordered[len(x_reordered)-1] - ref[0], 0)
#     delta_y = tf.where(ref[1] > edge_y, tf.abs(ref[1] - edge_y), 0)
#     area = delta_x.numpy()*delta_y.numpy()
#     tf_perato_area += area
#     total_points += tf.where(delta_y>0, 1, 0)
#     edge_y = new_edge_y

#     return tf_perato_area 


def hypervolume(points, reference_point):
    """
    Calculate the hypervolume of a set of 2D points with respect to a reference point.
    
    :param points: A 2D tensor of shape (N, 2) where N is the number of points.
    :param reference_point: A tensor of shape (2,) representing the reference point.
    :return: The hypervolume as a float tensor.
    """
    
    good_indices = np.where((points[:, 0] <= reference_point[0])&(points[:, 1]<= reference_point[1]))[0]
    points = tf.gather(points, good_indices)
    
    points = find_pareto_front_tf(points[:, 0], points[:, 1])
    
    # TODO: Improve to remove this if statement; currently the code breaks without this if statement if number of points are less than 3
    if points.shape[0] < 3:  # This if works as long as this function is not called multiple times in gradient tape
        return 0.
    # Calculate contributions of each point to the hypervolume
    # Calculate widths and heights relative to the reference point
    widths = reference_point[0] - points[:, 0]
    heights = tf.concat([reference_point[1:], tf.squeeze(points[:, 1][:-1])], axis=0) - points[:, 1]

    # Calculate hypervolume contributions (only positive contributions)
    contributions = tf.maximum(0.0, widths * heights)

    # Sum contributions to get total hypervolume
    total_hypervolume = tf.reduce_sum(contributions)

    return total_hypervolume