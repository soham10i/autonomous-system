
import numpy as np
import cv2
import math
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
import time
from collections import deque

@dataclass
class ExplorationMetrics:
    """Metrics for tracking exploration efficiency"""
    total_area_explored: float
    exploration_rate: float  # area per time
    coverage_percentage: float
    frontier_quality_score: float
    path_efficiency: float
    timestamp: float

@dataclass
class MapQualityMetrics:
    """Metrics for map quality assessment"""
    noise_level: float
    consistency_score: float
    feature_clarity: float
    occupancy_confidence: float
    update_frequency: float

class IntelligentFrontierSelector:
    """
    Advanced frontier selection with multi-criteria optimization.
    Improves exploration efficiency through intelligent target selection.
    """
    
    def __init__(self, occupancy_grid, min_frontier_size=8, exploration_memory_size=50):
        self.occupancy_grid = occupancy_grid
        self.min_frontier_size = min_frontier_size
        self.exploration_memory_size = exploration_memory_size
        
        # Exploration history and metrics
        self.explored_areas = []
        self.exploration_path = deque(maxlen=exploration_memory_size)
        self.frontier_history = {}
        self.area_coverage_map = None
        
        # Optimization parameters
        self.information_gain_weight = 0.35
        self.distance_weight = 0.25
        self.coverage_weight = 0.20
        self.accessibility_weight = 0.20
        
        # Coverage tracking
        self.last_coverage_update = 0
        self.coverage_update_interval = 5.0  # seconds
        
    def evaluate_frontier_information_gain(self, frontier_centroid, frontier_size):
        """
        Calculate potential information gain from exploring a frontier.
        Higher gain = more unknown area accessible.
        """
        x, y = frontier_centroid
        
        # Estimate viewable area from frontier position
        viewable_radius = 2.0  # Robot sensor range
        viewable_area = 0
        unknown_area = 0
        
        grid_x, grid_y = self.occupancy_grid.world_to_grid(x, y)
        view_radius_cells = int(viewable_radius / self.occupancy_grid.resolution)
        
        prob_grid = self.occupancy_grid.get_probability_grid()
        
        for dx in range(-view_radius_cells, view_radius_cells + 1):
            for dy in range(-view_radius_cells, view_radius_cells + 1):
                check_x, check_y = grid_x + dx, grid_y + dy
                
                if not self.occupancy_grid.is_valid_grid_coord(check_x, check_y):
                    continue
                
                # Check if within sensor range
                distance = math.sqrt(dx*dx + dy*dy) * self.occupancy_grid.resolution
                if distance > viewable_radius:
                    continue
                
                viewable_area += 1
                
                # Check if area is unknown (probability around 0.5)
                cell_prob = prob_grid[check_x, check_y]
                if 0.4 <= cell_prob <= 0.6:  # Unknown area
                    unknown_area += 1
        
        # Information gain is ratio of unknown to total viewable area
        if viewable_area == 0:
            return 0.0
        
        information_gain = unknown_area / viewable_area
        
        # Bonus for larger frontiers (more likely to lead to bigger areas)
        size_bonus = min(1.0, frontier_size / 20.0)
        
        return information_gain * (1.0 + 0.5 * size_bonus)
    
    def evaluate_coverage_efficiency(self, frontier_centroid, robot_position):
        """
        Evaluate how well this frontier fills coverage gaps.
        Prioritizes areas that haven't been explored recently.
        """
        x, y = frontier_centroid
        
        # Check coverage in the area around this frontier
        coverage_radius = 1.5  # meters
        grid_x, grid_y = self.occupancy_grid.world_to_grid(x, y)
        radius_cells = int(coverage_radius / self.occupancy_grid.resolution)
        
        coverage_score = 0.0
        total_cells = 0
        
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                check_x, check_y = grid_x + dx, grid_y + dy
                
                if not self.occupancy_grid.is_valid_grid_coord(check_x, check_y):
                    continue
                
                distance = math.sqrt(dx*dx + dy*dy) * self.occupancy_grid.resolution
                if distance > coverage_radius:
                    continue
                
                total_cells += 1
                
                # Check if this area has been visited recently
                world_pos = self.occupancy_grid.grid_to_world(check_x, check_y)
                recently_visited = False
                
                for past_pos in list(self.exploration_path)[-20:]:  # Last 20 positions
                    past_distance = math.sqrt(
                        (world_pos[0] - past_pos[0])**2 + 
                        (world_pos[1] - past_pos[1])**2
                    )
                    if past_distance < 0.8:  # Within 80cm of past position
                        recently_visited = True
                        break
                
                if not recently_visited:
                    coverage_score += 1.0
        
        return coverage_score / max(total_cells, 1)
    
    def evaluate_accessibility(self, frontier_centroid, robot_position):
        """
        Evaluate how accessible the frontier is for the robot.
        Considers obstacles and path complexity.
        """
        # Simple accessibility based on straight-line path
        x1, y1 = robot_position
        x2, y2 = frontier_centroid
        
        # Sample points along the line
        num_samples = 20
        obstacles_encountered = 0
        
        for i in range(num_samples):
            t = i / (num_samples - 1)
            sample_x = x1 + t * (x2 - x1)
            sample_y = y1 + t * (y2 - y1)
            
            grid_x, grid_y = self.occupancy_grid.world_to_grid(sample_x, sample_y)
            if self.occupancy_grid.is_valid_grid_coord(grid_x, grid_y):
                prob_grid = self.occupancy_grid.get_probability_grid()
                if prob_grid[grid_x, grid_y] > 0.7:  # Likely occupied
                    obstacles_encountered += 1
        
        # Accessibility decreases with more obstacles
        accessibility = max(0.0, 1.0 - (obstacles_encountered / num_samples))
        
        # Bonus for shorter distances
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distance_factor = max(0.1, 1.0 / (1.0 + distance / 3.0))  # Prefer closer targets
        
        return accessibility * distance_factor
    
    def select_optimal_frontier(self, frontiers, robot_position):
        """
        Select the optimal frontier using multi-criteria optimization.
        """
        if not frontiers:
            return None
        
        best_frontier = None
        best_score = -1.0
        
        for frontier in frontiers:
            centroid = frontier['centroid_world']
            size = frontier['size']
            
            # Calculate individual criteria scores
            info_gain = self.evaluate_frontier_information_gain(centroid, size)
            coverage_eff = self.evaluate_coverage_efficiency(centroid, robot_position)
            accessibility = self.evaluate_accessibility(centroid, robot_position)
            
            # Distance score (closer is better, but not too close)
            distance = math.sqrt(
                (centroid[0] - robot_position[0])**2 + 
                (centroid[1] - robot_position[1])**2
            )
            
            if distance < 0.5:
                distance_score = distance / 0.5  # Penalize very close
            elif distance <= 2.0:
                distance_score = 1.0  # Optimal range
            else:
                distance_score = max(0.1, 2.0 / distance)  # Penalize very far
            
            # Combined score using weights
            combined_score = (
                self.information_gain_weight * info_gain +
                self.coverage_weight * coverage_eff +
                self.accessibility_weight * accessibility +
                self.distance_weight * distance_score
            )
            
            # Apply exploration history penalty
            frontier_key = (int(centroid[0] * 10), int(centroid[1] * 10))  # 10cm grid
            if frontier_key in self.frontier_history:
                time_since_last = time.time() - self.frontier_history[frontier_key]
                if time_since_last < 30.0:  # Recently explored
                    combined_score *= 0.5
            
            if combined_score > best_score:
                best_score = combined_score
                best_frontier = frontier
        
        # Update frontier history
        if best_frontier:
            centroid = best_frontier['centroid_world']
            frontier_key = (int(centroid[0] * 10), int(centroid[1] * 10))
            self.frontier_history[frontier_key] = time.time()
        
        return best_frontier
    
    def update_exploration_metrics(self, robot_position):
        """Update exploration tracking and metrics."""
        current_time = time.time()
        
        # Add current position to exploration path
        self.exploration_path.append(robot_position)
        
        # Update coverage map periodically
        if current_time - self.last_coverage_update > self.coverage_update_interval:
            self._update_coverage_map()
            self.last_coverage_update = current_time
    
    def _update_coverage_map(self):
        """Update the coverage map based on exploration path."""
        prob_grid = self.occupancy_grid.get_probability_grid()
        coverage_map = np.zeros_like(prob_grid)
        
        for position in self.exploration_path:
            grid_x, grid_y = self.occupancy_grid.world_to_grid(position[0], position[1])
            
            if self.occupancy_grid.is_valid_grid_coord(grid_x, grid_y):
                # Mark explored area around robot position
                explore_radius = int(1.0 / self.occupancy_grid.resolution)  # 1 meter radius
                
                for dx in range(-explore_radius, explore_radius + 1):
                    for dy in range(-explore_radius, explore_radius + 1):
                        check_x, check_y = grid_x + dx, grid_y + dy
                        
                        if self.occupancy_grid.is_valid_grid_coord(check_x, check_y):
                            distance = math.sqrt(dx*dx + dy*dy) * self.occupancy_grid.resolution
                            if distance <= 1.0:
                                coverage_map[check_x, check_y] = 1.0
        
        self.area_coverage_map = coverage_map
    
    def get_exploration_metrics(self):
        """Get current exploration efficiency metrics."""
        if self.area_coverage_map is None:
            self._update_coverage_map()
        
        # Calculate metrics
        prob_grid = self.occupancy_grid.get_probability_grid()
        
        # Total explorable area (free + unknown space)
        explorable_mask = prob_grid < 0.8  # Not definitely occupied
        total_explorable = np.sum(explorable_mask)
        
        # Explored area
        if self.area_coverage_map is not None:
            explored_area = np.sum(self.area_coverage_map * explorable_mask)
            coverage_percentage = explored_area / max(total_explorable, 1) * 100.0
        else:
            coverage_percentage = 0.0
        
        # Exploration rate (area per time)
        time_span = len(self.exploration_path) * 0.1  # Assume 10Hz updates
        exploration_rate = coverage_percentage / max(time_span, 1.0)
        
        return ExplorationMetrics(
            total_area_explored=explored_area if self.area_coverage_map is not None else 0,
            exploration_rate=exploration_rate,
            coverage_percentage=coverage_percentage,
            frontier_quality_score=0.0,  # Would need frontier analysis
            path_efficiency=self._calculate_path_efficiency(),
            timestamp=time.time()
        )
    
    def _calculate_path_efficiency(self):
        """Calculate efficiency of exploration path."""
        if len(self.exploration_path) < 2:
            return 1.0
        
        # Calculate total path length
        total_distance = 0.0
        for i in range(1, len(self.exploration_path)):
            prev_pos = self.exploration_path[i-1]
            curr_pos = self.exploration_path[i]
            segment_length = math.sqrt(
                (curr_pos[0] - prev_pos[0])**2 + 
                (curr_pos[1] - prev_pos[1])**2
            )
            total_distance += segment_length
        
        # Calculate straight-line distance from start to current
        if len(self.exploration_path) >= 2:
            start_pos = self.exploration_path[0]
            end_pos = self.exploration_path[-1]
            straight_distance = math.sqrt(
                (end_pos[0] - start_pos[0])**2 + 
                (end_pos[1] - start_pos[1])**2
            )
            
            # Efficiency is inverse of path deviation
            if total_distance > 0:
                efficiency = min(1.0, straight_distance / total_distance + 0.1)
            else:
                efficiency = 1.0
        else:
            efficiency = 1.0
        
        return efficiency

class MapQualityEnhancer:
    """
    Enhanced map quality through noise filtering and intelligent updates.
    """
    
    def __init__(self, occupancy_grid, noise_threshold=0.1, consistency_window=5):
        self.occupancy_grid = occupancy_grid
        self.noise_threshold = noise_threshold
        self.consistency_window = consistency_window
        
        # Noise filtering parameters
        self.gaussian_kernel_size = 3
        self.median_kernel_size = 3
        self.bilateral_d = 5
        self.bilateral_sigma_color = 10
        self.bilateral_sigma_space = 10
        
        # Update tracking
        self.cell_update_counts = {}
        self.cell_consistency_scores = {}
        self.last_filter_time = 0
        self.filter_interval = 2.0  # Filter every 2 seconds
        
    def apply_noise_filtering(self):
        """
        Apply advanced noise filtering to the occupancy grid.
        """
        current_time = time.time()
        
        if current_time - self.last_filter_time < self.filter_interval:
            return
        
        prob_grid = self.occupancy_grid.get_probability_grid()
        
        # Apply multiple filtering techniques
        filtered_grid = self._apply_multi_stage_filtering(prob_grid)
        
        # Update the occupancy grid with filtered values
        self._update_grid_with_filtered_values(filtered_grid)
        
        self.last_filter_time = current_time
    
    def _apply_multi_stage_filtering(self, prob_grid):
        """Apply multi-stage filtering pipeline."""
        # Convert to 8-bit for OpenCV operations
        grid_8bit = (prob_grid * 255).astype(np.uint8)
        
        # Stage 1: Median filter to remove salt-and-pepper noise
        median_filtered = cv2.medianBlur(grid_8bit, self.median_kernel_size)
        
        # Stage 2: Gaussian filter for general smoothing
        gaussian_filtered = cv2.GaussianBlur(
            median_filtered, 
            (self.gaussian_kernel_size, self.gaussian_kernel_size), 
            0
        )
        
        # Stage 3: Bilateral filter to preserve edges while smoothing
        bilateral_filtered = cv2.bilateralFilter(
            gaussian_filtered,
            self.bilateral_d,
            self.bilateral_sigma_color,
            self.bilateral_sigma_space
        )
        
        # Convert back to probability space
        filtered_prob = bilateral_filtered.astype(np.float32) / 255.0
        
        # Stage 4: Consistency-based filtering
        consistency_filtered = self._apply_consistency_filtering(prob_grid, filtered_prob)
        
        return consistency_filtered
    
    def _apply_consistency_filtering(self, original_grid, filtered_grid):
        """Apply consistency-based filtering using update history."""
        result_grid = filtered_grid.copy()
        
        for i in range(original_grid.shape[0]):
            for j in range(original_grid.shape[1]):
                cell_key = (i, j)
                
                # Get update count for this cell
                update_count = self.cell_update_counts.get(cell_key, 0)
                
                # If cell has been updated many times, trust it more
                if update_count > 10:
                    # Blend original and filtered based on consistency
                    consistency = self.cell_consistency_scores.get(cell_key, 0.5)
                    blend_factor = min(0.8, consistency)
                    
                    result_grid[i, j] = (
                        blend_factor * original_grid[i, j] + 
                        (1 - blend_factor) * filtered_grid[i, j]
                    )
        
        return result_grid
    
    def _update_grid_with_filtered_values(self, filtered_grid):
        """Update occupancy grid with filtered values."""
        # Only update cells that have changed significantly
        prob_grid = self.occupancy_grid.get_probability_grid()
        
        for i in range(filtered_grid.shape[0]):
            for j in range(filtered_grid.shape[1]):
                original_prob = prob_grid[i, j]
                filtered_prob = filtered_grid[i, j]
                
                # Only update if change is significant
                if abs(original_prob - filtered_prob) > self.noise_threshold:
                    # Convert probability back to occupancy
                    if filtered_prob < 0.3:
                        occupied = False
                    elif filtered_prob > 0.7:
                        occupied = True
                    else:
                        continue  # Keep uncertain cells unchanged
                    
                    self.occupancy_grid.update_cell(i, j, occupied)
    
    def update_cell_consistency(self, grid_x, grid_y, measurement_confidence):
        """Update consistency tracking for a cell."""
        cell_key = (grid_x, grid_y)
        
        # Update count
        self.cell_update_counts[cell_key] = self.cell_update_counts.get(cell_key, 0) + 1
        
        # Update consistency score (exponential moving average)
        current_consistency = self.cell_consistency_scores.get(cell_key, 0.5)
        alpha = 0.1  # Learning rate
        new_consistency = (1 - alpha) * current_consistency + alpha * measurement_confidence
        self.cell_consistency_scores[cell_key] = new_consistency
    
    def enhance_map_features(self):
        """
        Enhance map features for better visual clarity and path planning.
        """
        prob_grid = self.occupancy_grid.get_probability_grid()
        
        # Enhance edges and boundaries
        enhanced_grid = self._enhance_boundaries(prob_grid)
        
        # Fill small holes in obstacles
        filled_grid = self._fill_small_holes(enhanced_grid)
        
        # Remove isolated noise points
        cleaned_grid = self._remove_isolated_points(filled_grid)
        
        return cleaned_grid
    
    def _enhance_boundaries(self, prob_grid):
        """Enhance obstacle boundaries for clearer mapping."""
        # Use morphological operations to enhance boundaries
        grid_8bit = (prob_grid * 255).astype(np.uint8)
        
        # Detect edges
        edges = cv2.Canny(grid_8bit, 50, 150)
        
        # Dilate edges slightly
        kernel = np.ones((3, 3), np.uint8)
        enhanced_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Combine with original
        enhanced_grid = grid_8bit.copy()
        enhanced_grid[enhanced_edges > 0] = 255  # Mark edges as definitely occupied
        
        return enhanced_grid.astype(np.float32) / 255.0
    
    def _fill_small_holes(self, prob_grid):
        """Fill small holes in obstacles."""
        grid_8bit = (prob_grid * 255).astype(np.uint8)
        
        # Create binary mask for obstacles
        obstacle_mask = grid_8bit > 180  # Definitely occupied
        
        # Fill small holes
        kernel = np.ones((3, 3), np.uint8)
        filled_mask = cv2.morphologyEx(obstacle_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # Apply back to grid
        result_grid = prob_grid.copy()
        result_grid[filled_mask > 0] = 0.9  # Mark filled areas as highly occupied
        
        return result_grid
    
    def _remove_isolated_points(self, prob_grid):
        """Remove isolated noise points."""
        grid_8bit = (prob_grid * 255).astype(np.uint8)
        
        # Find connected components
        _, labels, stats, _ = cv2.connectedComponentsWithStats(grid_8bit > 128, connectivity=8)
        
        # Remove small components
        min_component_size = 5  # pixels
        for label in range(1, len(stats)):  # Skip background (label 0)
            if stats[label, cv2.CC_STAT_AREA] < min_component_size:
                grid_8bit[labels == label] = 128  # Set to uncertain
        
        return grid_8bit.astype(np.float32) / 255.0
    
    def get_map_quality_metrics(self):
        """Calculate current map quality metrics."""
        prob_grid = self.occupancy_grid.get_probability_grid()
        
        # Noise level (variance in uncertain areas)
        uncertain_mask = (prob_grid > 0.3) & (prob_grid < 0.7)
        noise_level = np.std(prob_grid[uncertain_mask]) if np.any(uncertain_mask) else 0.0
        
        # Feature clarity (edge strength)
        grid_8bit = (prob_grid * 255).astype(np.uint8)
        edges = cv2.Canny(grid_8bit, 50, 150)
        feature_clarity = np.mean(edges) / 255.0
        
        # Occupancy confidence (percentage of certain cells)
        certain_mask = (prob_grid < 0.3) | (prob_grid > 0.7)
        occupancy_confidence = np.sum(certain_mask) / prob_grid.size
        
        # Consistency score (average of tracked cells)
        consistency_scores = list(self.cell_consistency_scores.values())
        consistency_score = np.mean(consistency_scores) if consistency_scores else 0.5
        
        return MapQualityMetrics(
            noise_level=noise_level,
            consistency_score=consistency_score,
            feature_clarity=feature_clarity,
            occupancy_confidence=occupancy_confidence,
            update_frequency=len(self.cell_update_counts) / max(1, time.time())
        )