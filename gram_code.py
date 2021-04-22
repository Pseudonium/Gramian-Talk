import numpy as np

def compute_parallelotope_volume(vectors):
    # Initially d-dimensional
    # Rotate down to n-dimensional
    # n is number of vectors we're given
    d = len(vectors[0])
    n = len(vectors)
    standard_basis = np.identity(d)
    
    orthonormal_span_basis = orthonormalize(make_basis(vectors))
    
    full_orthonormal_basis = orthonormalize(
        extend_to_basis(
            initial=orthonormal_span_basis, spanning_set=standard_basis
        )
    )
    
    orthogonal_matrix = change_of_basis_matrix(
        full_orthonormal_basis, standard_basis
    )
    
    vectors = [orthogonal_matrix * vector for vector in vectors]
    
    # Then, trim trailing zeroes
    vectors = [vector[:n] for vector in vectors] 
    
    return abs(np.linalg.det(vectors))