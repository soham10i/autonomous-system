
import numpy as np
import math
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging
from enum import Enum

import logging
import time
from slam_logging_config import get_slam_logger, performance_monitor, SLAMLogger

class SystemState(Enum):
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    CRITICAL = "critical"
    SHUTDOWN = "shutdown"

@dataclass
class SensorHealth:
    """Health status of individual sensors"""
    sensor_name: str
    is_active: bool = True
    last_update_time: float = 0.0
    error_count: int = 0
    consecutive_failures: int = 0
    health_score: float = 1.0  # [0.0, 1.0]
    status_message: str = "Initialized"

@dataclass
class SystemDiagnostics:
    def __init__(self):
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.update_frequency = 0.0
        self.error_count = 0
        self.warnings = []

    """Overall system health diagnostics"""
    system_state: SystemState
    uptime: float
    sensor_health: Dict[str, SensorHealth]
    memory_usage: float
    cpu_usage: float
    error_log: List[str]
    performance_metrics: Dict[str, float]

class DynamicOccupancyGrid:
    """
    Enhanced occupancy grid with dynamic sizing and adaptive resolution.
    Automatically expands to accommodate new areas and optimizes memory usage.
    """
    
    def __init__(self, initial_size=(100, 100), resolution=0.05, expansion_threshold=10, origin_x=0.0, origin_y=0.0):
        self.resolution = resolution
        self.expansion_threshold = expansion_threshold  # cells from edge before expansion
        
        # Grid storage
        self.grid_data = np.full(initial_size, 0.5, dtype=np.float32)  # log-odds
        self.probability_cache = None
        self.cache_valid = False
        
        # Grid coordinate system
        self.origin_x = origin_x  # World coordinates of grid origin
        self.origin_y = origin_y
        self.rows, self.cols = initial_size
        
        # Usage tracking
        self.active_cells = set()  # Set of (row, col) tuples with data
        self.bounds_min_x = float('inf')
        self.bounds_max_x = float('-inf')
        self.bounds_min_y = float('inf') 
        self.bounds_max_y = float('-inf')
        
        # Performance tracking
        self.update_count = 0
        self.expansion_count = 0
        self.last_optimization_time = time.time()
        
    def world_to_grid(self, world_x, world_y):
        """Convert world coordinates to grid indices."""
        grid_x = int((world_x - self.origin_x) / self.resolution)
        grid_y = int((world_y - self.origin_y) / self.resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid indices to world coordinates."""
        world_x = self.origin_x + (grid_x + 0.5) * self.resolution
        world_y = self.origin_y + (grid_y + 0.5) * self.resolution
        return world_x, world_y
    
    def is_valid_grid_coord(self, grid_x, grid_y):
        """Check if grid coordinates are valid."""
        return 0 <= grid_x < self.rows and 0 <= grid_y < self.cols
    
    def update_cell(self, world_x, world_y, occupied, confidence=1.0):
        """Update a cell with automatic grid expansion if needed."""
        try:
            # Check if expansion is needed
            self._check_and_expand_grid(world_x, world_y)
            
            # Convert to grid coordinates
            grid_x, grid_y = self.world_to_grid(world_x, world_y)
            
            if not self.is_valid_grid_coord(grid_x, grid_y):
                # Still invalid after expansion - this shouldn't happen
                logging.warning(f"Invalid grid coordinates after expansion: ({grid_x}, {grid_y})")
                return False
            
            # Update cell using log-odds
            if occupied:
                log_odds_update = math.log(0.9 / 0.1) * confidence  # Occupied evidence
            else:
                log_odds_update = math.log(0.1 / 0.9) * confidence  # Free evidence
            
            self.grid_data[grid_x, grid_y] += log_odds_update
            
            # Clamp to reasonable range
            self.grid_data[grid_x, grid_y] = np.clip(self.grid_data[grid_x, grid_y], -10.0, 10.0)
            
            # Track active cells and bounds
            self.active_cells.add((grid_x, grid_y))
            self._update_bounds(world_x, world_y)
            
            # Invalidate probability cache
            self.cache_valid = False
            self.update_count += 1
            
            return True
            
        except Exception as e:            
            self.logger.error(f"Error: {e}")
            logging.error(f"Error updating cell at ({world_x}, {world_y}): {e}")
            return False
    
    def _check_and_expand_grid(self, world_x, world_y):
        """Check if grid needs expansion and expand if necessary."""
        grid_x, grid_y = self.world_to_grid(world_x, world_y)
        
        # Check if we need to expand
        expansion_needed = False
        new_rows, new_cols = self.rows, self.cols
        offset_x, offset_y = 0, 0
        
        # Check boundaries
        if grid_x < self.expansion_threshold:
            # Expand left
            expansion_needed = True
            expand_amount = max(50, self.expansion_threshold - grid_x + 10)
            new_cols += expand_amount
            offset_y = expand_amount
            
        elif grid_x >= self.rows - self.expansion_threshold:
            # Expand right
            expansion_needed = True
            expand_amount = max(50, grid_x - self.rows + self.expansion_threshold + 10)
            new_rows += expand_amount
            
        if grid_y < self.expansion_threshold:
            # Expand down
            expansion_needed = True
            expand_amount = max(50, self.expansion_threshold - grid_y + 10)
            new_cols += expand_amount
            offset_x = expand_amount
            
        elif grid_y >= self.cols - self.expansion_threshold:
            # Expand up
            expansion_needed = True
            expand_amount = max(50, grid_y - self.cols + self.expansion_threshold + 10)
            new_cols += expand_amount
        
        if expansion_needed:
            self._expand_grid(new_rows, new_cols, offset_x, offset_y)
    
    def _expand_grid(self, new_rows, new_cols, offset_x, offset_y):
        """Expand the grid to new dimensions."""
        try:
            # Create new grid
            new_grid = np.full((new_rows, new_cols), 0.5, dtype=np.float32)
            
            # Copy old data
            old_start_x = offset_x
            old_end_x = offset_x + self.rows
            old_start_y = offset_y
            old_end_y = offset_y + self.cols
            
            new_grid[old_start_x:old_end_x, old_start_y:old_end_y] = self.grid_data
            
            # Update grid parameters
            self.grid_data = new_grid
            self.rows, self.cols = new_rows, new_cols
            
            # Update origin
            self.origin_x -= offset_x * self.resolution
            self.origin_y -= offset_y * self.resolution
            
            # Update active cells coordinates
            new_active_cells = set()
            for old_x, old_y in self.active_cells:
                new_x = old_x + offset_x
                new_y = old_y + offset_y
                new_active_cells.add((new_x, new_y))
            self.active_cells = new_active_cells
            
            self.expansion_count += 1
            self.cache_valid = False
            
            logging.info(f"Grid expanded to {new_rows}x{new_cols}, offset: ({offset_x}, {offset_y})")
            
        except Exception as e:
            self.logger.error(f"Error: {e}")
            logging.error(f"Error expanding grid: {e}")
    
    def _update_bounds(self, world_x, world_y):
        """Update the bounds of explored area."""
        self.bounds_min_x = min(self.bounds_min_x, world_x)
        self.bounds_max_x = max(self.bounds_max_x, world_x)
        self.bounds_min_y = min(self.bounds_min_y, world_y)
        self.bounds_max_y = max(self.bounds_max_y, world_y)
    
    def get_probability_grid(self):
        """Get probability grid, using cache if valid."""
        if self.cache_valid and self.probability_cache is not None:
            return self.probability_cache
        
        # Convert log-odds to probabilities
        self.probability_cache = 1.0 / (1.0 + np.exp(-self.grid_data))
        self.cache_valid = True
        
        return self.probability_cache
    
    def optimize_memory(self):
        """Optimize memory usage by trimming unused areas."""
        try:
            if len(self.active_cells) == 0:
                return
            
            # Find bounds of active area
            active_rows = [cell[0] for cell in self.active_cells]
            active_cols = [cell[1] for cell in self.active_cells]
            
            min_row = max(0, min(active_rows) - 20)  # Keep 20 cell buffer
            max_row = min(self.rows, max(active_rows) + 20)
            min_col = max(0, min(active_cols) - 20)
            max_col = min(self.cols, max(active_cols) + 20)
            
            # Only optimize if significant space can be saved
            new_rows = max_row - min_row
            new_cols = max_col - min_col
            
            if new_rows < self.rows * 0.8 or new_cols < self.cols * 0.8:
                # Trim the grid
                trimmed_grid = self.grid_data[min_row:max_row, min_col:max_col]
                
                # Update parameters
                self.grid_data = trimmed_grid
                self.rows, self.cols = new_rows, new_cols
                
                # Update origin
                self.origin_x += min_row * self.resolution
                self.origin_y += min_col * self.resolution
                
                # Update active cells
                new_active_cells = set()
                for old_x, old_y in self.active_cells:
                    new_x = old_x - min_row
                    new_y = old_y - min_col
                    if 0 <= new_x < new_rows and 0 <= new_y < new_cols:
                        new_active_cells.add((new_x, new_y))
                self.active_cells = new_active_cells
                
                self.cache_valid = False
                logging.info(f"Grid optimized to {new_rows}x{new_cols}")
                
        except Exception as e:            
            self.logger.error(f"Error: {e}")
            logging.error(f"Error optimizing grid memory: {e}")
    
    def update_grid(self, current_pose, lidar_data, camera_image=None):
        """Update the grid with LiDAR data and current pose."""
        if not lidar_data:
            return
        
        robot_x, robot_y, robot_theta = current_pose
        
        # Process LiDAR range data
        num_points = len(lidar_data)
        angle_increment = 2 * np.pi / num_points
        
        for i, distance in enumerate(lidar_data):
            if distance == float('inf') or distance > 10.0:  # Skip invalid readings
                continue
            
            # Calculate angle for this measurement
            angle = i * angle_increment + robot_theta
            
            # Calculate end point of the ray
            end_x = robot_x + distance * np.cos(angle)
            end_y = robot_y + distance * np.sin(angle)
            
            # Mark the end point as occupied
            self.update_cell(end_x, end_y, occupied=True, confidence=0.8)
            
            # Mark points along the ray as free (ray tracing)
            num_steps = int(distance / (self.resolution * 0.5))  # Sample every half resolution
            for step in range(1, num_steps):
                step_distance = step * distance / num_steps
                step_x = robot_x + step_distance * np.cos(angle)
                step_y = robot_y + step_distance * np.sin(angle)
                self.update_cell(step_x, step_y, occupied=False, confidence=0.3)
    
    def get_statistics(self):
        """Get grid statistics for monitoring."""
        active_percentage = len(self.active_cells) / (self.rows * self.cols) * 100
        memory_mb = self.grid_data.nbytes / (1024 * 1024)
        
        return {
            'grid_size': (self.rows, self.cols),
            'resolution': self.resolution,
            'active_cells': len(self.active_cells),
            'active_percentage': active_percentage,
            'memory_usage_mb': memory_mb,
            'update_count': self.update_count,
            'expansion_count': self.expansion_count,
            'explored_bounds': {
                'min_x': self.bounds_min_x,
                'max_x': self.bounds_max_x,
                'min_y': self.bounds_min_y,
                'max_y': self.bounds_max_y
            }
        }

class ErrorHandler:
    """
    Comprehensive error handling and recovery system for SLAM components.
    """
    
    def __init__(self, max_error_history=100):
        self.max_error_history = max_error_history
        self.error_history = []
        self.sensor_health = {}
        self.recovery_strategies = {}
        
        # Error thresholds
        self.max_consecutive_failures = 5
        self.health_recovery_time = 10.0  # seconds
        self.critical_error_threshold = 0.2  # 20% health score
        
        # Initialize sensor health tracking
        self._initialize_sensor_health()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('SLAM_ErrorHandler')
    
    def _initialize_sensor_health(self):
        """Initialize health tracking for all sensors."""
        sensors = ['camera', 'lidar', 'imu', 'encoders', 'visual_odometry', 'slam']
        
        for sensor in sensors:
            self.sensor_health[sensor] = SensorHealth(
                sensor_name=sensor,
                is_active=True,
                last_update_time=time.time(),
                error_count=0,
                consecutive_failures=0,
                health_score=1.0,
                status_message="Initialized"
            )
    
    def report_sensor_update(self, sensor_name, success=True, error_message=""):
        """Report a sensor update (success or failure)."""
        current_time = time.time()
        
        if sensor_name not in self.sensor_health:
            self._initialize_sensor_health()
        
        health = self.sensor_health[sensor_name]
        health.last_update_time = current_time
        
        if success:
            # Successful update
            health.consecutive_failures = 0
            health.is_active = True
            health.status_message = "Operating normally"
            
            # Gradually improve health score
            health.health_score = min(1.0, health.health_score + 0.1)
            
        else:
            # Failed update
            health.error_count += 1
            health.consecutive_failures += 1
            health.status_message = error_message or "Update failed"
            
            # Reduce health score
            health.health_score = max(0.0, health.health_score - 0.2)
            
            # Deactivate sensor if too many failures
            if health.consecutive_failures >= self.max_consecutive_failures:
                health.is_active = False
                health.status_message = f"Deactivated after {health.consecutive_failures} failures"
                self.logger.warning(f"Sensor {sensor_name} deactivated due to consecutive failures")
            
            # Log error
            self._log_error(sensor_name, error_message)
    
    def _log_error(self, sensor_name, error_message):
        """Log an error to the error history."""
        error_entry = {
            'timestamp': time.time(),
            'sensor': sensor_name,
            'message': error_message,
            'system_state': self._assess_system_state()
        }
        
        self.error_history.append(error_entry)
        
        # Maintain history size
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)
        
        self.logger.error(f"Sensor {sensor_name}: {error_message}")
    
    def check_sensor_timeout(self, sensor_name, timeout_duration=5.0):
        """Check if a sensor has timed out (no updates for too long)."""
        if sensor_name not in self.sensor_health:
            return False
        
        health = self.sensor_health[sensor_name]
        time_since_update = time.time() - health.last_update_time
        
        if time_since_update > timeout_duration and health.is_active:
            self.report_sensor_update(sensor_name, False, f"Timeout after {time_since_update:.1f}s")
            return True
        
        return False
    
    def attempt_sensor_recovery(self, sensor_name):
        """Attempt to recover a failed sensor."""
        if sensor_name not in self.sensor_health:
            return False
        
        health = self.sensor_health[sensor_name]
        current_time = time.time()
        
        # Only attempt recovery if enough time has passed
        if current_time - health.last_update_time < self.health_recovery_time:
            return False
        
        try:
            # Attempt recovery (sensor-specific strategies would go here)
            self.logger.info(f"Attempting recovery for sensor {sensor_name}")
            
            # Reset consecutive failures and reactivate
            health.consecutive_failures = 0
            health.is_active = True
            health.health_score = 0.5  # Start with moderate health
            health.status_message = "Recovery attempted"
            health.last_update_time = current_time
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error: {e}")
            self.logger.error(f"Recovery failed for sensor {sensor_name}: {e}")
            return False
    
    def _assess_system_state(self):
        """Assess overall system state based on sensor health."""
        if not self.sensor_health:
            return SystemState.INITIALIZING
        
        active_sensors = sum(1 for h in self.sensor_health.values() if h.is_active)
        total_sensors = len(self.sensor_health)
        avg_health = np.mean([h.health_score for h in self.sensor_health.values()])
        
        # Determine system state
        if active_sensors == 0:
            return SystemState.CRITICAL
        elif active_sensors < total_sensors * 0.5:
            return SystemState.DEGRADED
        elif avg_health < self.critical_error_threshold:
            return SystemState.RECOVERING
        else:
            return SystemState.HEALTHY
    
    def get_diagnostics(self):
        """Get comprehensive system diagnostics."""
        current_time = time.time()
        
        # Calculate performance metrics
        recent_errors = [e for e in self.error_history if current_time - e['timestamp'] < 60.0]
        error_rate = len(recent_errors) / 60.0  # errors per second
        
        return SystemDiagnostics(
            system_state=self._assess_system_state(),
            uptime=current_time - min(h.last_update_time for h in self.sensor_health.values()),
            sensor_health=self.sensor_health.copy(),
            memory_usage=0.0,  # Would be calculated from actual memory usage
            cpu_usage=0.0,     # Would be calculated from actual CPU usage
            error_log=[e['message'] for e in self.error_history[-10:]],
            performance_metrics={
                'error_rate': error_rate,
                'active_sensors': sum(1 for h in self.sensor_health.values() if h.is_active),
                'avg_health_score': np.mean([h.health_score for h in self.sensor_health.values()])
            }
        )
    
    def safe_execute(self, function, *args, sensor_name=None, **kwargs):
        """Safely execute a function with automatic error handling."""
        try:
            result = function(*args, **kwargs)
            if sensor_name:
                self.report_sensor_update(sensor_name, True)
            return result, None
            
        except Exception as e:            
            self.logger.error(f"Error: {e}")
            error_msg = f"Function {function.__name__} failed: {str(e)}"
            if sensor_name:
                self.report_sensor_update(sensor_name, False, error_msg)
            else:
                self._log_error("system", error_msg)
            return None, e

class PerformanceMonitor:
    """
    Performance monitoring and optimization for real-time SLAM operation.
    """
    
    def __init__(self, target_fps=10.0):
        self.target_fps = target_fps
        self.target_frame_time = 1.0 / target_fps
        
        # Performance tracking
        self.frame_times = []
        self.max_frame_history = 100
        self.component_times = {}
        
        # Optimization flags
        self.enable_adaptive_quality = True
        self.enable_dynamic_resolution = True
        self.performance_mode = "balanced"  # "speed", "balanced", "quality"
        
    def start_frame(self):
        """Start timing a new frame."""
        self.frame_start_time = time.time()
        self.component_times = {}
    
    def start_component(self, component_name):
        """Start timing a component within the frame."""
        self.component_times[component_name] = {'start': time.time()}
    
    def end_component(self, component_name):
        """End timing a component."""
        if component_name in self.component_times:
            self.component_times[component_name]['duration'] = time.time() - self.component_times[component_name]['start']
    
    def end_frame(self):
        """End frame timing and update statistics."""
        frame_duration = time.time() - self.frame_start_time
        self.frame_times.append(frame_duration)
        
        # Maintain history
        if len(self.frame_times) > self.max_frame_history:
            self.frame_times.pop(0)
        
        return frame_duration
    
    def get_performance_stats(self):
        """Get current performance statistics."""
        if not self.frame_times:
            return {}
        
        recent_times = self.frame_times[-20:]  # Last 20 frames
        avg_frame_time = np.mean(recent_times)
        current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        return {
            'current_fps': current_fps,
            'target_fps': self.target_fps,
            'avg_frame_time': avg_frame_time,
            'target_frame_time': self.target_frame_time,
            'performance_ratio': self.target_frame_time / avg_frame_time if avg_frame_time > 0 else 0,
            'frame_time_std': np.std(recent_times),
            'component_times': self.component_times.copy()
        }
    
    def should_reduce_quality(self):
        """Determine if quality should be reduced to maintain performance."""
        if not self.enable_adaptive_quality or len(self.frame_times) < 10:
            return False
        
        recent_avg = np.mean(self.frame_times[-10:])
        return recent_avg > self.target_frame_time * 1.2  # 20% over target
    
    def should_increase_quality(self):
        """Determine if quality can be increased."""
        if not self.enable_adaptive_quality or len(self.frame_times) < 10:
            return False
        
        recent_avg = np.mean(self.frame_times[-10:])
        return recent_avg < self.target_frame_time * 0.8  # 20% under target
    
    def optimize_parameters(self):
        """Suggest optimization parameters based on current performance."""
        stats = self.get_performance_stats()
        
        if stats['performance_ratio'] < 0.8:  # Running too slow
            return {
                'reduce_grid_resolution': True,
                'reduce_feature_count': True,
                'increase_keyframe_distance': True,
                'simplify_planning': True
            }
        elif stats['performance_ratio'] > 1.2:  # Running too fast, can increase quality
            return {
                'increase_grid_resolution': False,
                'increase_feature_count': False,
                'decrease_keyframe_distance': False,
                'enhance_planning': False
            }
        else:
            return {}  # No changes needed

class RobustSLAMManager:
    """
    Main robustness manager that coordinates all robustness components.
    """
    
    def __init__(self, initial_grid_size=(100, 100), resolution=0.05, loop_closure_detector=None):
        # Core components
        self.dynamic_grid = DynamicOccupancyGrid(
            initial_size=initial_grid_size,
            resolution=resolution,
            origin_x=-(initial_grid_size[0] * resolution) / 2,
            origin_y=-(initial_grid_size[1] * resolution) / 2
        )
        self.error_handler = ErrorHandler()
        self.slam_logger = get_slam_logger("RobustSLAMManager")
        self.loop_closure_detector = loop_closure_detector
        
        # State management
        self.is_initialized = False
        self.startup_time = time.time()
        self.last_optimization_time = time.time()
        self.optimization_interval = 30.0  # seconds
        
        # Configuration
        self.auto_recovery_enabled = True
        self.performance_optimization_enabled = True
        
    def initialize(self):
        """Initialize the robust SLAM system."""
        try:
            self.slam_logger.info("Initializing robust SLAM system...")
            
            # Initialize components
            self.error_handler._initialize_sensor_health()
            
            self.is_initialized = True
            self.slam_logger.info("Robust SLAM system initialized successfully")
            return True
            
        except Exception as e:          
            self.slam_logger.error(f"Failed to initialize robust SLAM system: {e}")
            return False
    
    @performance_monitor
    def update_system(self, sensor_data, current_pose):
        """Main system update with robustness features."""
        if not self.is_initialized:
            return None
        
        # Start performance monitoring
        # self.performance_monitor.start_frame()
        
        try:
            # Process sensor data with error handling
            processed_data = self._process_sensor_data_safely(sensor_data)
            
            # Update grid
            # self.performance_monitor.start_component("grid_update")
            self._update_grid_safely(processed_data, current_pose)
            # self.performance_monitor.end_component("grid_update")

            # Perform loop closure detection
            if self.loop_closure_detector and "camera" in processed_data and "lidar" in processed_data:
                # self.performance_monitor.start_component("loop_closure_detection")
                loop_closure = self.loop_closure_detector.detect_loop_closure(
                    current_pose, processed_data["camera"], processed_data["lidar"]
                )
                if loop_closure:
                    self.slam_logger.info(f"Loop closure detected: {loop_closure}")
                    # Here you would typically trigger a global optimization/correction
                # self.performance_monitor.end_component("loop_closure_detection")
            
            # Periodic optimization
            current_time = time.time()
            if current_time - self.last_optimization_time > self.optimization_interval:
                self._perform_maintenance()
                self.last_optimization_time = current_time
            
            # End performance monitoring
            # frame_time = self.performance_monitor.end_frame()
            
            return self._get_system_status()
            
        except Exception as e:         
            self.slam_logger.error(f"System update failed: {e}")
            self.error_handler._log_error("system", f"System update failed: {e}")
            return None
    
    def _process_sensor_data_safely(self, sensor_data):
        """Process sensor data with error handling and timeout checking."""
        processed = {}
        
        # Check for sensor timeouts
        for sensor_name in ['camera', 'lidar', 'encoders']:
            self.error_handler.check_sensor_timeout(sensor_name)
        
        # Process each sensor type safely
        if 'lidar' in sensor_data:
            result, error = self.error_handler.safe_execute(
                self._process_lidar_data, 
                sensor_data['lidar'], 
                sensor_name='lidar'
            )
            if result is not None:
                processed['lidar'] = result
        
        if 'camera' in sensor_data:
            result, error = self.error_handler.safe_execute(
                self._process_camera_data,
                sensor_data['camera'],
                sensor_name='camera'
            )
            if result is not None:
                processed['camera'] = result
        
        return processed
    
    def _process_lidar_data(self, lidar_data):
        """Process LiDAR data (placeholder for actual processing)."""
        # This would contain actual LiDAR processing logic
        return lidar_data
    
    def _process_camera_data(self, camera_data):
        """Process camera data (placeholder for actual processing)."""
        # This would contain actual camera processing logic
        return camera_data
    
    def _update_grid_safely(self, processed_data, current_pose):
        """Update occupancy grid with error handling."""
        if 'lidar' in processed_data:
            lidar_data = processed_data['lidar']
            camera_image = processed_data.get('camera')
            self.error_handler.safe_execute(
                self.dynamic_grid.update_grid,
                current_pose,
                lidar_data,
                camera_image,
                sensor_name='occupancy_grid'
            )
    
    def _perform_maintenance(self):
        """Perform periodic system maintenance."""
        try:
            # Optimize grid memory
            self.dynamic_grid.optimize_memory()
            
            # Attempt sensor recovery if needed
            if self.auto_recovery_enabled:
                for sensor_name, health in self.error_handler.sensor_health.items():
                    if not health.is_active:
                        self.error_handler.attempt_sensor_recovery(sensor_name)
            
            # Performance optimization
            if self.performance_optimization_enabled:
                # optimizations = self.performance_monitor.optimize_parameters()
                # self._apply_optimizations(optimizations)
                pass
                
        except Exception as e:      
            self.slam_logger.error(f"Maintenance failed: {e}")
            self.error_handler._log_error("maintenance", f"Maintenance failed: {e}")
    
    def _apply_optimizations(self, optimizations):
        """Apply performance optimizations."""
        # This would implement actual optimization changes
        pass
    
    def _get_system_status(self):
        """Get comprehensive system status."""
        return {
            'diagnostics': self.error_handler.get_diagnostics(),
            'grid_stats': self.dynamic_grid.get_statistics(),
            'uptime': time.time() - self.startup_time,
            'is_healthy': self.error_handler._assess_system_state() == SystemState.HEALTHY
        }
    
    def get_grid(self):
        """Get the dynamic occupancy grid."""
        return self.dynamic_grid
    
    def shutdown(self):
        """Gracefully shutdown the system."""
        self.logger.info("Shutting down robust SLAM system...")
        # Cleanup code would go here
        self.is_initialized = False
