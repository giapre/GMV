import json
import numpy as np

with open("cortical_GAMLSS_coeffs.json", "r") as f:
    models = json.load(f)
print(models)

# Function to get BCTo parameters
def bcto_parameters(sex, subregion, age):
    coefs = models[sex][subregion]
    
    # Handle missing coefficients 
    def safe_exp(val):
        return np.exp(val) if val is not None else np.nan
    
    def safe_val(val):
        return val if val is not None else np.nan
    
    mu    = safe_exp(coefs['mu_intercept']) + safe_val(coefs['mu_age']) * age if coefs['mu_intercept'] is not None else np.nan
    sigma = safe_exp(coefs['sigma_intercept']) + safe_val(coefs['sigma_age']) * age if coefs['sigma_intercept'] is not None else np.nan
    nu    = safe_val(coefs['nu_intercept']) + safe_val(coefs['nu_age']) * age
    tau   = safe_exp(coefs['tau_intercept']) + safe_val(coefs['tau_age']) * age if coefs['tau_intercept'] is not None else np.nan
    
    return mu, sigma, nu, tau

# Function to compute Z-score
def bcto_z(y, sex, subregion, age):
    mu, sigma, nu, tau = bcto_parameters(sex, subregion, age)
    
    if np.isnan(mu) or np.isnan(sigma) or np.isnan(nu):
        return np.nan  # Cannot compute z-score if parameters missing
    
    if abs(nu) > 1e-6:
        z = ((y / mu)**nu - 1) / (nu * sigma)
    else:
        z = np.log(y / mu) / sigma
    return z

# Example usage
# z_score = bcto_z(y=2.5, sex='male', subregion='Region_1', age=25)
# print(z_score)