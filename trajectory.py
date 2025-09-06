import numpy as np

def golden_spiral_trajectory(config):
    num_images = config["camera"]["num_images"]
    phi_g = np.pi*(3 - np.sqrt(5))
    theta_min = np.deg2rad(config["camera"]["theta_min_deg"])
    theta_max = np.deg2rad(config["camera"]["theta_max_deg"])
    
    phi_start_deg = config["camera"].get("phi_start_deg", 0)
    phi_end_deg = config["camera"].get("phi_end_deg", 360)
    phi_start_rad = np.deg2rad(phi_start_deg)
    phi_end_rad = np.deg2rad(phi_end_deg)
    
    phi_range_rad = phi_end_rad - phi_start_rad
    if phi_range_rad < 2*np.pi:
        phi_g = phi_range_rad / num_images
    else:
        phi_g = np.pi*(3 - np.sqrt(5))

    i = np.arange(num_images)
    z = np.cos(theta_min) + i/(num_images-1)*(np.cos(theta_max) - np.cos(theta_min))
    thetas = np.arccos(z)
    
    if phi_range_rad < 2*np.pi:
        phis = phi_start_rad + i * phi_g
    else:
        phis = (i * phi_g) % (2*np.pi)  
    
    return {
        "mode": "spherical",
        "thetas": thetas,
        "phis": phis,
        "rotation_step": phi_g,
        "look_at": "center"
    }

def diamond_trajectory(config):
    num_images = config["camera"]["num_images"]
    plane_distance = config["camera"].get("plane_distance", 2.0)
    diamond_size = config["camera"].get("diamond_size", 1.0)
    look_at = config["camera"].get("look_at", "perpendicular")
    trajectory_offset = config["camera"].get("trajectory_offset", [0, 0, 0])
    
    t = np.linspace(0, 1, num_images)
    
    x = np.zeros(num_images)
    y = np.zeros(num_images)
    
    for i, ti in enumerate(t):
        if ti <= 0.25:
            x[i] = 4 * ti * diamond_size
            y[i] = 0
        elif ti <= 0.5:
            x[i] = diamond_size
            y[i] = 4 * (ti - 0.25) * diamond_size
        elif ti <= 0.75:
            x[i] = diamond_size - 4 * (ti - 0.5) * diamond_size
            y[i] = diamond_size
        else:
            x[i] = 0
            y[i] = diamond_size - 4 * (ti - 0.75) * diamond_size
    
    z = np.full(num_images, plane_distance)
    
    positions = np.stack([x, y, z], axis=-1)
    positions += np.array(trajectory_offset)
    
    return {
        "mode": "planar",
        "positions": positions,
        "plane_normal": config["camera"].get("plane_normal", [0, 1, 0]),
        "rotation_step": 0,
        "look_at": look_at,
        "trajectory_offset": trajectory_offset
    }

def zigzag_trajectory(config):
    num_images = config["camera"]["num_images"]
    plane_distance = config["camera"].get("plane_distance", 2.0)
    zigzag_width = config["camera"].get("zigzag_width", 1.0)
    zigzag_height = config["camera"].get("zigzag_height", 1.0)
    look_at = config["camera"].get("look_at", "perpendicular")
    trajectory_offset = config["camera"].get("trajectory_offset", [0, 0, 0])
    
    t = np.linspace(0, 1, num_images)
    
    x = np.zeros(num_images)
    y = np.zeros(num_images)
    
    for i, ti in enumerate(t):
        if ti <= 0.33:
            x[i] = 3 * ti * zigzag_width
            y[i] = 0
        elif ti <= 0.66:
            x[i] = zigzag_width - 3 * (ti - 0.33) * zigzag_width
            y[i] = 3 * (ti - 0.33) * zigzag_height
        else:
            x[i] = 3 * (ti - 0.66) * zigzag_width
            y[i] = zigzag_height
    
    z = np.full(num_images, plane_distance)
    
    positions = np.stack([x, y, z], axis=-1)
    positions += np.array(trajectory_offset)
    
    return {
        "mode": "planar",
        "positions": positions,
        "plane_normal": config["camera"].get("plane_normal", [0, 1, 0]),
        "rotation_step": 0,
        "look_at": look_at,
        "trajectory_offset": trajectory_offset
    }

def plus_trajectory(config):
    num_images = config["camera"]["num_images"]
    plane_distance = config["camera"].get("plane_distance", 2.0)
    plus_horizontal_length = config["camera"].get("plus_horizontal_length", 1.0)
    plus_vertical_length = config["camera"].get("plus_vertical_length", 1.0)
    look_at = config["camera"].get("look_at", "perpendicular")
    trajectory_offset = config["camera"].get("trajectory_offset", [0, 0, 0])
    
    t = np.linspace(0, 1, num_images)
    
    x = np.zeros(num_images)
    y = np.zeros(num_images)
    
    for i, ti in enumerate(t):
        if ti <= 0.5:
            x[i] = (ti - 0.25) * 2 * plus_horizontal_length
            y[i] = 0
        else:
            x[i] = 0
            y[i] = (ti - 0.75) * 2 * plus_vertical_length
    
    z = np.full(num_images, plane_distance)
    
    positions = np.stack([x, y, z], axis=-1)
    positions += np.array(trajectory_offset)
    
    return {
        "mode": "planar",
        "positions": positions,
        "plane_normal": config["camera"].get("plane_normal", [0, 1, 0]),
        "rotation_step": 0,
        "look_at": look_at,
        "trajectory_offset": trajectory_offset
    }


def get_trajectory(config):
    trajectory_type = config["camera"].get("trajectory_type", "golden_spiral")
    
    if trajectory_type == "golden_spiral":
        return golden_spiral_trajectory(config)
    elif trajectory_type == "diamond":
        return diamond_trajectory(config)
    elif trajectory_type == "zigzag":
        return zigzag_trajectory(config)
    elif trajectory_type == "plus":
        return plus_trajectory(config)
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")
