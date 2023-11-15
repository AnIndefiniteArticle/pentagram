# non-dimensional bending angle (theta_hat = 1 at half-light)
def ftheta_hat(theta_hat,x):
    return np.log(theta_hat)+theta_hat - 1. - x
