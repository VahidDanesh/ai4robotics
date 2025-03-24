import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import ListedColormap


def plot_robot(q, obstacles=None, L1=3, L2=3):
    """
    Plot a 2R planar robot configuration with optional obstacles
    
    Args:
        q: List of joint angles [q1, q2] in radians
        obstacles: List of obstacles, each defined by a list of (x, y) vertices
        L1: Length of first link (default=3)
        L2: Length of second link (default=3)
    """
    # Calculate joint positions
    q1, q2 = q
    x1 = L1 * np.cos(q1)
    y1 = L1 * np.sin(q1)
    x2 = x1 + L2 * np.cos(q1 + q2)
    y2 = y1 + L2 * np.sin(q1 + q2)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot obstacles
    if obstacles:
        for obstacle in obstacles:
            poly = Polygon(obstacle, closed=True, 
                            edgecolor='black', 
                            fill=True,
                            facecolor='gray', 
                            linewidth=2)
            ax.add_patch(poly)

    # Plot robot
    ax.plot([0, x1, x2], [0, y1, y2], 'bo-', linewidth=3, markersize=10)
    ax.plot(0, 0, 'ko', markersize=10)  # Base joint
    
    # Configure plot
    ax.set_xlim(-(L1+L2)*1.2, (L1+L2)*1.2)
    ax.set_ylim(-(L1+L2)*1.2, (L1+L2)*1.2)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title(f'2R Planar Robot: q1={np.rad2deg(q1):.1f}°, q2={np.rad2deg(q2):.1f}°')
    plt.show()


def plot_configuration_space(obstacles=None, L1=3, L2=3, step_deg=1):
    """
    Plot configuration space for a 2R planar robot
    
    Args:
        obstacles: List of obstacles (each as list of (x,y) vertices)
        L1: Length of first link (default=3)
        L2: Length of second link (default=3)
        step_deg: Angular resolution in degrees (default=1°)
    """
    # Helper functions
    def ccw(A, B, C):
        return (B[0]-A[0])*(C[1]-A[1]) - (B[1]-A[1])*(C[0]-A[0])
    
    def segments_intersect(A, B, C, D):
        return ccw(A,C,D) * ccw(B,C,D) < 0 and ccw(A,B,C) * ccw(A,B,D) < 0
    
    def point_in_polygon(p, polygon):
        x, y = p
        inside = False
        n = len(polygon)
        p1x, p1y = polygon[0]
        for i in range(n+1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    # Generate configuration grid
    step = np.deg2rad(step_deg)
    q1 = np.arange(0, 2*np.pi + step, step)
    q2 = np.arange(0, 2*np.pi + step, step)
    cspace = np.zeros((len(q2), len(q1)))

    # Check collisions for each configuration
    for i, ang1 in enumerate(q1):
        for j, ang2 in enumerate(q2):
            # Forward kinematics
            x1 = L1*np.cos(ang1)
            y1 = L1*np.sin(ang1)
            x2 = x1 + L2*np.cos(ang1 + ang2)
            y2 = y1 + L2*np.sin(ang1 + ang2)
            
            # Check collision for both links
            collision = False
            if obstacles:
                for poly in obstacles:
                    # Check first link (base to joint1)
                    if point_in_polygon((0, 0), poly) or point_in_polygon((x1, y1), poly):
                        collision = True
                        break
                    # Check second link (joint1 to end-effector)
                    if point_in_polygon((x1, y1), poly) or point_in_polygon((x2, y2), poly):
                        collision = True
                        break
                    # Check edge intersections
                    for k in range(len(poly)):
                        v1 = poly[k]
                        v2 = poly[(k+1)%len(poly)]
                        if segments_intersect((0, 0), (x1, y1), v1, v2):
                            collision = True
                            break
                        if segments_intersect((x1, y1), (x2, y2), v1, v2):
                            collision = True
                            break
            cspace[j, i] = 1 if collision else 0

    # Plot configuration space
    plt.figure(figsize=(8, 8))
    plt.imshow(cspace, origin='lower', extent=[0, 360, 0, 360],
               cmap=ListedColormap(['black', 'white']), aspect='auto')
    plt.xlabel('q1 (degrees)')
    plt.ylabel('q2 (degrees)')
    plt.title(f'Configuration Space (Resolution: {step_deg}°)')
    plt.grid(True, alpha=0.3)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Sample configuration 
    q = [0.25*np.pi, 1*np.pi]  
    
    obstacles = [
        [(3.1, 2.2), (5.2, 2.6), (4.5, 4.0)],       
        [(-2.4, 2.2), (-0.4, 2.7), (-1.0, 4.0)],
        [(-2.4, -3.2), (-0.5, -2.8), (-1.0, -1.2)]  
    ]
    
    plot_robot(q, obstacles)
    # plot_configuration_space(obstacles, step_deg=2)
