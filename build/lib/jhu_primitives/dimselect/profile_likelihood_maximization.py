def profile_likelihood_maximization(U, n_elbows, threshold):
    """
    Inputs
        U - An ordered or unordered list of eigenvalues
        n - The number of elbows to return

    Return
        elbows - A numpy array containing elbows
    """
    if type(U) == list: # cast to array for functionality later
        U = np.array(U)
    
    if type(U) is not np.ndarray: # only support arrays, lists
        return np.array([])
    
    if n_elbows == 0: # nothing to do..
        return np.array([])
    
    if U.ndim == 2:
        U = np.std(U, axis = 0)
    
    U = U[U > threshold]
    
    if len(U) == 0:
        return np.array([])
    
    elbows = []
    
    if len(U) == 1:
        return np.array(elbows.append(U[0]))
    
    # select values greater than the threshold
    U.sort() # sort
    U = U[::-1] # reverse array so that it is sorted in descending order
    
    while len(elbows) < n_elbows and len(U) > 1:
        d = 1
        sample_var = np.var(U, ddof = 1)
        sample_scale = sample_var**(1/2)
        elbow = 0
        likelihood_elbow = 0
        while d < len(U):
            mean_sig = np.mean(U[:d])
            mean_noise = np.mean(U[d:])
            sig_likelihood = 0
            noise_likelihood = 0
            for i in range(d):
                sig_likelihood += norm.pdf(U[i], mean_sig, sample_scale)
            for i in range(d, len(U)):
                noise_likelihood += norm.pdf(U[i], mean_noise, sample_scale)
            
            likelihood = noise_likelihood + sig_likelihood
        
            if likelihood > likelihood_elbow:
                likelihood_elbow = likelihood 
                elbow = d
            d += 1
        elbows.append(U[elbow - 1])
        U = U[elbow:]
        
    if len(elbows) == n_elbows:
        return np.array(elbows)
    
    if len(U) == 0:
        return np.array(elbows)
    else:
        elbows.append(U[0])
        return np.array(elbows)