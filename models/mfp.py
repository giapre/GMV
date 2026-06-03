import json

# Load the combined JSON file
with open("MFP_thickness.json", "r") as f:
    mfp_models = json.load(f)

def compute_region_zscore(region_name: str, sex: str, age: float, mean_thickness: float):
    """
    Compute z-score for a given brain region, sex, and patient info.

    Parameters:
        region_name (str): Name of the brain region (matches JSON key in male/female list).
        sex (str): 'female' or 'male'.
        age (float): Patient's age.
        mean_thickness (float): Observed mean thickness for the region.

    Returns:
        float: z-score
    """
    sex = sex.lower()
    if sex not in mfp_models:
        raise ValueError(f"Sex '{sex}' not found in models.")
    
    # Each sex is a list of regions; find the region
    region_model = None
    for reg in range(len(mfp_models[sex])):
        if region_name == reg:
            region_model = mfp_models[sex][region_name]
            break
    if region_model is None:
        raise ValueError(f"Region '{region_name}' not found in {sex} models.")
    
    # Compute predicted thickness
    predicted = 0.0
    for key, value in region_model.items():
        coeff = value['coefficients']
        if key.lower() == 'intercept':
            predicted += coeff
        elif key.startswith("age"):
            # Fractional polynomial term: age.1 -> age^1, age.2 -> age^2
            try:
                power = int(key.split(".")[1])
                predicted += coeff * (age ** power)
            except Exception:
                predicted += coeff * age
        else:
            # Any other term for this region, just add coefficient (for simple models)
            predicted += coeff

    # Compute z-score
    z = (mean_thickness - predicted) / mean_thickness
    return z

