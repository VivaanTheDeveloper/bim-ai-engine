import numpy as np

def resolve_annotation_collisions(raw_wall_lines, predicted_annotation_coords, safe_buffer_mm=600):
    """
    Mathematical physics-vector check. Evaluates AI text placement choices 
    and micro-adjusts coordinates to prevent line-on-text overlaps.
    """
    adjusted_coordinates = []
    
    for label_pt in predicted_annotation_coords:
        lx, ly = label_pt[0], label_pt[1]
        
        for wall in raw_wall_lines:
            (wx1, wy1), (wx2, wy2) = wall[0], wall[1]
            
            # Calculate the minimum perpendicular mathematical distance from point to line segment
            wall_vector = np.array([wx2 - wx1, wy2 - wy1])
            point_vector = np.array([lx - wx1, ly - wy1])
            
            line_len_sq = dot_product = np.dot(wall_vector, wall_vector)
            if line_len_sq == 0:
                continue
                
            t = max(0, min(1, np.dot(point_vector, wall_vector) / line_len_sq))
            projection_pt = np.array([wx1, wy1]) + t * wall_vector
            distance = np.linalg.norm(np.array([lx, ly]) - projection_pt)
            
            # COLLISION TRIGGER DETECTED: Adjust coordinate offset safely away from entity
            if distance < safe_buffer_mm:
                # Calculate normal vector projection to execute a clean linear shift
                if np.linalg.norm(wall_vector) != 0:
                    normal_vector = np.array([-wall_vector[1], wall_vector[0]])
                    normal_vector = normal_vector / np.linalg.norm(normal_vector)
                    
                    # Relocate target point coordinate vector outside safety threshold buffer zone
                    lx += normal_vector[0] * (safe_buffer_mm - distance + 100)
                    ly += normal_vector[1] * (safe_buffer_mm - distance + 100)
                    
        adjusted_coordinates.append([lx, ly])
        
    return adjusted_coordinates

if __name__ == "__main__":
    pass