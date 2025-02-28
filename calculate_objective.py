import numpy as np

def calculate_objective(scene_points, model_points, p_O):
    """
    Calculate the value of the objective function:
    sum_{i=0}^n || W_p_m_i - W_p_s_i ||²
    
    Args:
        scene_points: List of scene points in world frame (W_p_si)
        model_points: List of model points in object frame (O_p_mi)
        p_O: Translation vector (in this case [0,0])
        
    Returns:
        objective_value: The value of the objective function
    """
    # Identity rotation
    R_O = np.eye(2)
    
    # Initialize objective value
    objective_value = 0
    
    for i, (scene_p, model_p) in enumerate(zip(scene_points, model_points)):
        # Transform model point to world frame
        transformed_model_p = R_O @ np.array(model_p) + np.array(p_O)
        
        # Calculate squared distance
        squared_distance = np.sum((np.array(scene_p) - transformed_model_p)**2)
        
        # Add to objective value
        objective_value += squared_distance
    
    return objective_value

# Define scene points (W_p_si)
scene_points = [
    (1, 5),
    (3, 10),
    (5, 10)
]

# Define model points (O_p_mi)
model_points = [
    (-2, -5),
    (0, 0),
    (2, 0)
]

# Calculate objective value for p_O = (0,0)
p_O = (3, 10)
result = calculate_objective(scene_points, model_points, p_O)

print(f"The value of the objective function for p^O = {p_O} is: {result}")
