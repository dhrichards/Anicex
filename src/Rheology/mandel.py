from ufl import as_tensor, sqrt



def tensor2mandel(T):

    if T.ufl_shape == (2, 2):
        return as_tensor([T[0, 0], T[1, 1], sqrt(2)*T[0, 1]])
    
    if T.ufl_shape == (3, 3):
        return as_tensor([T[0, 0], T[1, 1], T[2, 2],
                 sqrt(2)*T[0, 1], sqrt(2)*T[1, 2], sqrt(2)*T[0, 2]])
    

    if T.ufl_shape == (2, 2, 2, 2):  # Fourth-order 2D symmetric tensor
        return as_tensor([[T[0,0,0,0], T[0,0,1,1], sqrt(2)*T[0,0,0,1]],
                          [T[1,1,0,0], T[1,1,1,1], sqrt(2)*T[1,1,0,1]],
                          [sqrt(2)*T[0,1,0,0], sqrt(2)*T[0,1,1,1], 2*T[0,1,0,1]]])
    
    if T.ufl_shape == (3, 3, 3, 3):  # Fourth-order 3D symmetric tensor
        return as_tensor([[T[0,0,0,0],          T[0,0,1,1],         T[0,0,2,2],   sqrt(2)*T[0,0,1,2], sqrt(2)*T[0,0,0,2], sqrt(2)*T[0,0,0,1]],
                          [T[1,1,0,0],          T[1,1,1,1],         T[1,1,2,2],   sqrt(2)*T[1,1,1,2], sqrt(2)*T[1,1,0,2], sqrt(2)*T[1,1,0,1]],
                          [T[2,2,0,0],          T[2,2,1,1],         T[2,2,2,2],   sqrt(2)*T[2,2,1,2], sqrt(2)*T[2,2,0,2], sqrt(2)*T[2,2,0,1]],
                  [sqrt(2)*T[1,2,0,0],  sqrt(2)*T[1,2,1,1], sqrt(2)*T[1,2,2,2],         2*T[1,2,1,2],       2*T[1,2,0,2],       2*T[1,2,0,1]],
                  [sqrt(2)*T[0,2,0,0],  sqrt(2)*T[0,2,1,1], sqrt(2)*T[0,2,2,2],         2*T[0,2,1,2],       2*T[0,2,0,2],       2*T[0,2,0,1]],
                  [sqrt(2)*T[0,1,0,0],  sqrt(2)*T[0,1,1,1], sqrt(2)*T[0,1,2,2],         2*T[0,1,1,2],       2*T[0,1,0,2],       2*T[0,1,0,1]]])
    

    else:
        raise ValueError("Invalid tensor shape")
    



def mandel2tensor(M):
    """
    Convert a 6x6 Mandel notation matrix to a 3x3x3x3 tensor.

    Parameters:
        mandel_array (list[list[float]]): A 6x6 array-like object in Mandel notation.

    Returns:
        ufl.Tensor: A 3x3x3x3 tensor.
    """
    if M.ufl_shape == (3,):
        return as_tensor([[M[0], M[2]/sqrt(2)], 
                          [M[2]/sqrt(2), M[1]]])
    
    if M.ufl_shape == (6,):
        return as_tensor([[M[0], M[5]/sqrt(2), M[4]/sqrt(2)],
                          [M[5]/sqrt(2), M[1], M[3]/sqrt(2)],
                          [M[4]/sqrt(2), M[3]/sqrt(2), M[2]]])
    
    else:


        if M.ufl_shape[0] == 6:
            D = 3
            mandel_to_tensor_map = {
                                    0: (0, 0),
                                    1: (1, 1),
                                    2: (2, 2),
                                    3: (1, 2),
                                    4: (0, 2),
                                    5: (0, 1),
                                }
        elif M.ufl_shape[0] == 3:
            D = 2
            mandel_to_tensor_map = {
                                    0: (0, 0),
                                    1: (1, 1),
                                    2: (0, 1),
                                }
        else:
            raise ValueError("Invalid Mandel array shape.")
        

        # Create an empty tensor represented as nested UFL expressions
        tensor = [[[[0 for _ in range(D)] for _ in range(D)] for _ in range(D)] for _ in range(D)]

        # Fill the 3x3x3x3 tensor using Mandel indices
        for i in range(M.ufl_shape[0]):
            for j in range(M.ufl_shape[1]):
                idx1 = mandel_to_tensor_map[i]
                idx2 = mandel_to_tensor_map[j]

                # Calculate the value for the tensor entry, accounting for symmetry factors
                factor = 1.0
                if i >= D:
                    factor *= sqrt(2)
                if j >= D:
                    factor *= sqrt(2)

                value = M[i][j] / factor

                # Directly write to all symmetric entries
                tensor[idx1[0]][idx1[1]][idx2[0]][idx2[1]] = value
                tensor[idx1[1]][idx1[0]][idx2[0]][idx2[1]] = value
                tensor[idx1[0]][idx1[1]][idx2[1]][idx2[0]] = value
                tensor[idx1[1]][idx1[0]][idx2[1]][idx2[0]] = value

        # Convert to UFL tensor
        return as_tensor(tensor)


