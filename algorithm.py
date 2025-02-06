# Algorithm 1

import numpy as np
import scipy
from itertools import groupby

# Find a list of subspaces, each one is a list of vectors corresponding to degenerate eigevalues of H
def list_of_subspaces(H, round_decimal = 8):
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    eigenvector_list = [np.reshape(eigenvectors[:, j], (eigenvectors.shape[0], 1)) for j in range(eigenvectors.shape[1])]
    subspace_index_list = [list(grp) for k, grp in groupby(np.around(eigenvalues, round_decimal))]
    num_subspaces = len(subspace_index_list)
    num_vectors_list = [len(subspace_index) for subspace_index in subspace_index_list]
    subspace_list = []

    # no need to include the last subspace
    for i in range(num_subspaces-1):
        subspace_of_vectors = []
        for j in range(num_vectors_list[i]):
            first_vector = eigenvector_list.pop(0)
            subspace_of_vectors.append(first_vector)
        subspace_list.append(subspace_of_vectors)
    return subspace_list

# Projection on the subspace
def Projection(subspace):
    P = 0
    for i in range(len(subspace)):
        P += subspace[i] @ subspace[i].conj().T
    return P

# updated list of subspaces with lower dimensions, by reducing the dimensions of two subspaces
def updated_list_of_subspaces(subspace1, subspace2, round_decimal = 8):
    stop = False
    orthogonality = False
    equivalence = False
    list_of_subspaces = []
    P1 = Projection(subspace1)
    P2 = Projection(subspace2)
    U, S, Vh = np.linalg.svd(P1 @ P2)
    S = np.around(S, round_decimal)

    num_nonzero_singular_values = len([ele for ele in S if not np.allclose(ele, 0)])
    
    S1 = S[:len(subspace1)]
    S2 = S[:len(subspace2)]

    subspace1_vector_list = [np.reshape(U[:, i], (U.shape[0], 1)) for i in range(num_nonzero_singular_values)]
    subspace2_vector_list = [np.reshape(Vh[i].conj(), (Vh.shape[1], 1)) for i in range(num_nonzero_singular_values)]

    if len(subspace1) > num_nonzero_singular_values:
        vals, vecs = np.linalg.eigh(P1 - Projection(subspace1_vector_list))
        for j in range(1, len(subspace1)-num_nonzero_singular_values+1):
            subspace1_vector_list.append(np.reshape(vecs[:, -j], (P1.shape[0], 1)))

    if len(subspace2) > num_nonzero_singular_values:
        vals, vecs = np.linalg.eigh(P2 - Projection(subspace2_vector_list))
        for j in range(1, len(subspace2)-num_nonzero_singular_values+1):
            subspace2_vector_list.append(np.reshape(vecs[:, -j], (P2.shape[0], 1)))
    
    subspace1_index_list = [list(grp) for k, grp in groupby(S1)]
    subspace2_index_list = [list(grp) for k, grp in groupby(S2)]
    num_subspaces1 = len(subspace1_index_list)
    num_subspaces2 = len(subspace2_index_list)
    num_vectors_list1 = [len(subspace_index) for subspace_index in subspace1_index_list]
    num_vectors_list2 = [len(subspace_index) for subspace_index in subspace2_index_list]
    
    # check whether all effective singular values are equal
    if all(np.allclose(x, S[0]) for x in S[:max(len(subspace1), len(subspace2))]):
        stop = True
        subspace1_of_vectors = subspace1
        list_of_subspaces.append(subspace1_of_vectors)
        if np.allclose(S[0], 0):
            orthogonality = True
            subspace2_of_vectors = subspace2
            list_of_subspaces.append(subspace2_of_vectors)
        elif np.allclose(S[0], 1.0):
            equivalence = True
            subspace2_of_vectors = subspace1
        else:
            U_ = np.hstack(subspace1_of_vectors)
            Vh_ = np.linalg.pinv(np.diag(S1)) @ U_.conj().T @ P1 @ P2
            subspace2_of_vectors = [np.reshape(Vh_[i].conj(), (Vh_.shape[1], 1)) for i in range(len(S1))]
            list_of_subspaces.append(subspace2_of_vectors)
 
    
    else:
        for i in range(num_subspaces1):
            subspace1_of_vectors = []
            for j in range(num_vectors_list1[i]):
                first_vector1 = subspace1_vector_list.pop(0)
                subspace1_of_vectors.append(first_vector1)
            list_of_subspaces.append(subspace1_of_vectors)
    
        for i in range(num_subspaces2):
            subspace2_of_vectors = []
            for j in range(num_vectors_list2[i]):
                first_vector2 = subspace2_vector_list.pop(0)
                subspace2_of_vectors.append(first_vector2)
            if not (i==0 and np.allclose(S[0], 1.0)):
                list_of_subspaces.append(subspace2_of_vectors)
    return list_of_subspaces, stop, orthogonality, equivalence

# Remove equivalent subspaces
def remove_equivalent_subspaces(list_of_subspaces):
    indices_to_remove = []
    num_subspaces = len(list_of_subspaces)
    for i in range(num_subspaces-1):
        for j in range(i+1, num_subspaces):
            list_of_subspaces_ij, stop_ij, orthogonality_ij, equivalence_ij = updated_list_of_subspaces(list_of_subspaces[i], list_of_subspaces[j])
            if equivalence_ij == True:
                indices_to_remove.append(j)
  
    for index in sorted(list(set(indices_to_remove)), reverse=True):
        del list_of_subspaces[index]
    return list_of_subspaces

# Reduce the dimension of subspaces, given a list of subspaces
def subspace_reduction(list_of_subspaces, round_decimal=8):
    if len(list_of_subspaces) == 1:
        return list_of_subspaces
    stop_condition = False
    
    # remove equivalent subspaces
    list_of_subspaces = remove_equivalent_subspaces(list_of_subspaces)
    
    while (not stop_condition):
        output_list_of_subspaces = [list_of_subspaces[0]]
        stop_list = []
        for i in range(1, len(list_of_subspaces)):
            list_of_subspaces_i =[]
            for j in range(len(output_list_of_subspaces)):
                list_of_subspaces_ij, stop_ij, orthogonality_ij, equivalence_ij = updated_list_of_subspaces(list_of_subspaces[i], output_list_of_subspaces[j])
                stop_list.append(stop_ij)
                list_of_subspaces_i += list_of_subspaces_ij
            output_list_of_subspaces = list_of_subspaces_i
            
            # remove equivalent subspaces
            output_list_of_subspaces = remove_equivalent_subspaces(output_list_of_subspaces)
        list_of_subspaces = output_list_of_subspaces
        stop_condition = all(stop == True for stop in stop_list)
    
    # remove equivalent subspaces
    list_of_subspaces = remove_equivalent_subspaces(list_of_subspaces)
    

    # reorganize subspaces
    num_subspaces = len(list_of_subspaces)
    reorganized_list_of_subspaces = []
    while len(reorganized_list_of_subspaces) < num_subspaces:
        reorganized_list_of_subspaces.append(list_of_subspaces[0])
        list_of_subspaces.pop(0)
        current_reorganized_list_of_subspaces = reorganized_list_of_subspaces
        current_list_of_subspaces = list_of_subspaces
        if len(list_of_subspaces) >= 1:
            indices_to_remove = []
            for i in range(len(current_reorganized_list_of_subspaces)):
                for j in range(len(current_list_of_subspaces)):
                    reorganized_subspace = current_reorganized_list_of_subspaces[i]
                    subspace = current_list_of_subspaces[j]
                    possible_list_of_subspaces, stop, orthogonality, equivalence = updated_list_of_subspaces(reorganized_subspace, subspace)
                    if orthogonality == False:
                        reorganized_list_of_subspaces.append(possible_list_of_subspaces[-1])
                        indices_to_remove.append(j)
            for index in sorted(list(set(indices_to_remove)), reverse=True):
                del list_of_subspaces[index]
        
    # remove equivalent subspaces
    reorganized_list_of_subspaces = remove_equivalent_subspaces(reorganized_list_of_subspaces)
    return reorganized_list_of_subspaces

# Find a unitary control that inverses the effect of T_theta
def unitary_control(H_in_list, H_out_list, round_decimal=8):
    list_all_subspaces_in = []
    list_all_subspaces_out = []
    for H_in in H_in_list:
        list_all_subspaces_in += list_of_subspaces(H_in)

    for H_out in H_out_list:
        list_all_subspaces_out += list_of_subspaces(H_out)
    
    reduced_list_of_subspace_in = subspace_reduction(list_all_subspaces_in)
    reduced_list_of_subspace_out = subspace_reduction(list_all_subspaces_out)
    reduced_list_of_vectors_in = [x for xs in reduced_list_of_subspace_in for x in xs]
    reduced_list_of_vectors_out = [x for xs in reduced_list_of_subspace_out for x in xs]
    Matrix_in = np.hstack(reduced_list_of_vectors_in)
    Matrix_out = np.hstack(reduced_list_of_vectors_out)
    M = Matrix_out @ Matrix_in.conj().T
    U, S, Vh = np.linalg.svd(M)
    unitary = U @ Vh
    return unitary.conj().T