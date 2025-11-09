#!/usr/bin/env python3
"""
Professional Visualization and Monitoring Components
Real-time OpenCV visualization with performance optimization
Author: Visualization Systems Engineer - October 2025
"""

import time
import numpy as np
import cv2
import math
from typing import Dict, List, Optional, Tuple, Any
import threading
import queue

from core_types import (
    Point2D, RobotPose, Pillar, PassageSegment, 
    PerformanceProfiler, SystemConfig
)
from interfaces import IVisualization, IDataLogger

class RealTimeVisualizer(IVisualization):
    """High-performance real-time visualization system

    Features:
    - Multi-threaded rendering for smooth display
    - Memory-efficient drawing operations
    - Comprehensive HUD with statistics
    - Performance monitoring
    """

    def __init__(self, window_name: str = "Autonomous Exploration", 
                 window_size: Tuple[int, int] = (800, 800),
                 profiler: Optional[PerformanceProfiler] = None):
        self.window_name = window_name
        self.window_size = window_size
        self.profiler = profiler

        # Visualization parameters
        self.grid_scale_factor = 1.0
        self.robot_size = 8
        self.pillar_size = 12
        self.path_thickness = 2

        # Color scheme (BGR format for OpenCV)
        self.colors = {
            'background': (64, 64, 64),
            'free_space': (255, 255, 255),
            'occupied': (0, 0, 0),
            'unknown': (128, 128, 128),
            'robot': (0, 0, 255),           # Red robot
            'robot_direction': (0, 255, 0), # Green direction arrow
            'blue_pillar': (255, 0, 0),     # Blue pillar
            'yellow_pillar': (0, 255, 255), # Yellow pillar
            'current_path': (255, 0, 255),  # Magenta path
            'target': (0, 255, 255),        # Cyan target
            'passage': (0, 255, 0),         # Green passages
            'text': (255, 255, 255)         # White text
        }

        # Threading for smooth display
        self.enable_threading = True
        self.display_queue = queue.Queue(maxsize=5)
        self.display_thread = None
        self.is_running = False

        # Current visualization state
        self.current_frame = None
        self.frame_count = 0
        self.last_update_time = 0.0

        # Performance tracking
        self.render_times = []
        self.frame_rates = []

        # Initialize OpenCV window
        self._init_display()

        print("📺 Real-time visualizer initialized")
        print(f"   Window: {window_name} ({window_size[0]}x{window_size[1]})")
        print(f"   Threading: {self.enable_threading}")

    def _init_display(self) -> None:
        """Initialize OpenCV display window"""
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.resizeWindow(self.window_name, self.window_size[0], self.window_size[1])

            # Create initial black frame
            initial_frame = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)
            cv2.putText(initial_frame, "Initializing Visualization...", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors['text'], 2)
            cv2.imshow(self.window_name, initial_frame)
            cv2.waitKey(1)

            # Start display thread if threading enabled
            if self.enable_threading:
                self.is_running = True
                self.display_thread = threading.Thread(target=self._display_worker)
                self.display_thread.daemon = True
                self.display_thread.start()

        except Exception as e:
            print(f"Display initialization error: {e}")

    def update_display(self, occupancy_grid: np.ndarray, robot_pose: RobotPose,
                      pillars: List[Pillar], path: Optional[List[Point2D]] = None,
                      **kwargs) -> None:
        """Update visualization display with current system state"""
        start_time = time.time()

        try:
            # Create visualization frame
            frame = self._create_visualization_frame(
                occupancy_grid, robot_pose, pillars, path, **kwargs
            )

            if frame is not None:
                if self.enable_threading:
                    # Add to display queue (non-blocking)
                    try:
                        self.display_queue.put(frame, block=False)
                    except queue.Full:
                        # Skip frame if queue full (maintains real-time performance)
                        pass
                else:
                    # Direct display
                    cv2.imshow(self.window_name, frame)
                    cv2.waitKey(1)

                self.current_frame = frame
                self.frame_count += 1

            # Performance tracking
            render_time = time.time() - start_time
            self.render_times.append(render_time)

            # Calculate frame rate
            current_time = time.time()
            if self.last_update_time > 0:
                frame_rate = 1.0 / (current_time - self.last_update_time)
                self.frame_rates.append(frame_rate)
            self.last_update_time = current_time

            # Keep only recent performance data
            if len(self.render_times) > 100:
                self.render_times = self.render_times[-100:]
            if len(self.frame_rates) > 100:
                self.frame_rates = self.frame_rates[-100:]

            if self.profiler:
                self.profiler.record_timing("visualization_render", render_time)

        except Exception as e:
            print(f"Visualization update error: {e}")

    def _display_worker(self) -> None:
        """Background thread worker for smooth display"""
        while self.is_running:
            try:
                # Get frame from queue with timeout
                frame = self.display_queue.get(timeout=0.1)

                # Display frame
                cv2.imshow(self.window_name, frame)
                cv2.waitKey(1)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Display worker error: {e}")
                break

    def _create_visualization_frame(self, occupancy_grid: np.ndarray, robot_pose: RobotPose,
                                  pillars: List[Pillar], path: Optional[List[Point2D]] = None,
                                  **kwargs) -> Optional[np.ndarray]:
        """Create complete visualization frame"""
        try:
            # Create base map visualization
            map_vis = self._create_map_visualization(occupancy_grid)

            if map_vis is None:
                return None

            # Resize to window size
            map_vis = cv2.resize(map_vis, self.window_size, interpolation=cv2.INTER_NEAREST)

            # Calculate scaling factors
            if occupancy_grid.size > 0:
                grid_height, grid_width = occupancy_grid.shape
                scale_x = self.window_size[0] / grid_width
                scale_y = self.window_size[1] / grid_height
                self.grid_scale_factor = min(scale_x, scale_y)

            # Draw robot
            self._draw_robot(map_vis, robot_pose)

            # Draw pillars
            self._draw_pillars(map_vis, pillars)

            # Draw path
            if path:
                self._draw_path(map_vis, path)

            # Draw passages
            passages = kwargs.get('passages', [])
            self._draw_passages(map_vis, passages)

            # Draw target
            target = kwargs.get('target')
            if target:
                self._draw_target(map_vis, target)

            # Add information overlay
            self._add_info_overlay(map_vis, robot_pose, pillars, **kwargs)

            return map_vis

        except Exception as e:
            print(f"Frame creation error: {e}")
            return None

    def _create_map_visualization(self, occupancy_grid: np.ndarray) -> Optional[np.ndarray]:
        """Create base map visualization from occupancy grid"""
        if occupancy_grid is None or occupancy_grid.size == 0:
            # Create empty visualization
            vis = np.full((400, 400, 3), self.colors['background'], dtype=np.uint8)
            cv2.putText(vis, "No Map Data", (150, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors['text'], 2)
            return vis

        height, width = occupancy_grid.shape

        # Create RGB visualization
        vis = np.zeros((height, width, 3), dtype=np.uint8)

        # Vectorized color mapping
        free_mask = (occupancy_grid == 0)
        occupied_mask = (occupancy_grid == 100)
        uncertain_mask = (occupancy_grid == 50)
        unknown_mask = (occupancy_grid == -1)

        vis[free_mask] = self.colors['free_space']
        vis[occupied_mask] = self.colors['occupied']
        vis[uncertain_mask] = self.colors['unknown']
        vis[unknown_mask] = self.colors['background']

        # Flip for proper display orientation
        return cv2.flip(vis, 0)

    def _draw_robot(self, frame: np.ndarray, robot_pose: RobotPose) -> None:
        """Draw robot position and orientation"""
        try:
            # Calculate robot position in image coordinates
            robot_x = int(robot_pose.x * self.grid_scale_factor + self.window_size[0] / 2)
            robot_y = int(-robot_pose.y * self.grid_scale_factor + self.window_size[1] / 2)

            # Check bounds
            if (0 <= robot_x < self.window_size[0] and 
                0 <= robot_y < self.window_size[1]):

                # Draw robot body
                cv2.circle(frame, (robot_x, robot_y), self.robot_size, 
                          self.colors['robot'], -1)

                # Draw orientation arrow
                arrow_length = self.robot_size * 2
                end_x = int(robot_x + arrow_length * math.cos(-robot_pose.theta))
                end_y = int(robot_y + arrow_length * math.sin(-robot_pose.theta))

                cv2.arrowedLine(frame, (robot_x, robot_y), (end_x, end_y),
                               self.colors['robot_direction'], 2)

        except Exception as e:
            print(f"Robot drawing error: {e}")

    def _draw_pillars(self, frame: np.ndarray, pillars: List[Pillar]) -> None:
        """Draw detected pillars"""
        try:
            for pillar in pillars:
                # Calculate pillar position
                pillar_x = int(pillar.position.x * self.grid_scale_factor + self.window_size[0] / 2)
                pillar_y = int(-pillar.position.y * self.grid_scale_factor + self.window_size[1] / 2)

                # Check bounds
                if (0 <= pillar_x < self.window_size[0] and 
                    0 <= pillar_y < self.window_size[1]):

                    # Select color based on pillar color
                    if pillar.color == 'blue':
                        color = self.colors['blue_pillar']
                    elif pillar.color == 'yellow':
                        color = self.colors['yellow_pillar']
                    else:
                        color = self.colors['text']

                    # Draw pillar with confidence-based size
                    pillar_size = int(self.pillar_size * (0.5 + 0.5 * pillar.confidence))
                    cv2.circle(frame, (pillar_x, pillar_y), pillar_size, color, -1)

                    # Draw confidence ring
                    confidence_radius = int(pillar_size * 1.5)
                    cv2.circle(frame, (pillar_x, pillar_y), confidence_radius, color, 2)

        except Exception as e:
            print(f"Pillar drawing error: {e}")

    def _draw_path(self, frame: np.ndarray, path: List[Point2D]) -> None:
        """Draw current path"""
        try:
            if len(path) < 2:
                return

            # Convert path to image coordinates
            path_points = []
            for point in path:
                x = int(point.x * self.grid_scale_factor + self.window_size[0] / 2)
                y = int(-point.y * self.grid_scale_factor + self.window_size[1] / 2)

                if (0 <= x < self.window_size[0] and 0 <= y < self.window_size[1]):
                    path_points.append((x, y))

            # Draw path segments
            for i in range(len(path_points) - 1):
                cv2.line(frame, path_points[i], path_points[i + 1],
                        self.colors['current_path'], self.path_thickness)

            # Draw waypoint circles
            for point in path_points[::5]:  # Every 5th point
                cv2.circle(frame, point, 3, self.colors['current_path'], -1)

        except Exception as e:
            print(f"Path drawing error: {e}")

    def _draw_passages(self, frame: np.ndarray, passages: List[PassageSegment]) -> None:
        """Draw detected passages"""
        try:
            for passage in passages[-5:]:  # Show recent passages only
                # Convert points to image coordinates
                start_x = int(passage.start_point.x * self.grid_scale_factor + self.window_size[0] / 2)
                start_y = int(-passage.start_point.y * self.grid_scale_factor + self.window_size[1] / 2)
                end_x = int(passage.end_point.x * self.grid_scale_factor + self.window_size[0] / 2)
                end_y = int(-passage.end_point.y * self.grid_scale_factor + self.window_size[1] / 2)

                # Check bounds
                if (0 <= start_x < self.window_size[0] and 0 <= start_y < self.window_size[1] and
                    0 <= end_x < self.window_size[0] and 0 <= end_y < self.window_size[1]):

                    cv2.line(frame, (start_x, start_y), (end_x, end_y),
                            self.colors['passage'], 2)

        except Exception as e:
            print(f"Passage drawing error: {e}")

    def _draw_target(self, frame: np.ndarray, target: Point2D) -> None:
        """Draw current exploration target"""
        try:
            target_x = int(target.x * self.grid_scale_factor + self.window_size[0] / 2)
            target_y = int(-target.y * self.grid_scale_factor + self.window_size[1] / 2)

            if (0 <= target_x < self.window_size[0] and 0 <= target_y < self.window_size[1]):
                cv2.circle(frame, (target_x, target_y), 10, self.colors['target'], 3)

        except Exception as e:
            print(f"Target drawing error: {e}")

    def _add_info_overlay(self, frame: np.ndarray, robot_pose: RobotPose, 
                         pillars: List[Pillar], **kwargs) -> None:
        """Add information overlay with system statistics"""
        try:
            # Setup text parameters
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            line_height = 25
            y_offset = 30

            # Title
            cv2.putText(frame, "AUTONOMOUS EXPLORATION SYSTEM", 
                       (10, y_offset), font, 0.8, self.colors['text'], 2)
            y_offset += line_height + 10

            # System state
            state = kwargs.get('state', 'Unknown')
            cv2.putText(frame, f"State: {state}", 
                       (10, y_offset), font, font_scale, self.colors['text'], thickness)
            y_offset += line_height

            # Time and performance
            exploration_time = kwargs.get('exploration_time', 0)
            cv2.putText(frame, f"Time: {exploration_time:.0f}s", 
                       (10, y_offset), font, font_scale, self.colors['text'], thickness)
            y_offset += line_height

            # Coverage and distance
            coverage = kwargs.get('coverage', 0)
            distance = kwargs.get('distance', 0)
            cv2.putText(frame, f"Coverage: {coverage:.1f}% | Distance: {distance:.1f}m", 
                       (10, y_offset), font, font_scale, self.colors['text'], thickness)
            y_offset += line_height

            # Pillars found
            pillars_found = len([p for p in pillars if p.confidence > 0.5])
            cv2.putText(frame, f"Pillars: {pillars_found}/2", 
                       (10, y_offset), font, font_scale, self.colors['text'], thickness)
            y_offset += line_height

            # Robot pose
            cv2.putText(frame, f"Pose: ({robot_pose.x:.2f}, {robot_pose.y:.2f}, {math.degrees(robot_pose.theta):.0f}°)", 
                       (10, y_offset), font, 0.5, self.colors['text'], thickness)
            y_offset += line_height

            # Performance stats
            if self.frame_rates:
                avg_fps = np.mean(self.frame_rates[-10:])  # Last 10 frames
                cv2.putText(frame, f"FPS: {avg_fps:.1f}", 
                           (10, y_offset), font, font_scale, self.colors['text'], thickness)
            y_offset += line_height

            # Legend (bottom of screen)
            legend_y = self.window_size[1] - 60
            cv2.putText(frame, "WHITE=Free | BLACK=Occupied | GRAY=Unknown", 
                       (10, legend_y), font, 0.5, self.colors['text'], thickness)
            legend_y += 20
            cv2.putText(frame, "RED=Robot | BLUE=Blue Pillar | YELLOW=Yellow Pillar | MAGENTA=Path", 
                       (10, legend_y), font, 0.5, self.colors['text'], thickness)

        except Exception as e:
            print(f"Info overlay error: {e}")

    def save_visualization(self, filename: str) -> bool:
        """Save current visualization to file"""
        try:
            if self.current_frame is not None:
                success = cv2.imwrite(filename, self.current_frame)
                if success:
                    print(f"📸 Visualization saved: {filename}")
                return success
            return False

        except Exception as e:
            print(f"Save visualization error: {e}")
            return False

    def close(self) -> None:
        """Close visualization system"""
        try:
            self.is_running = False

            if self.display_thread and self.display_thread.is_alive():
                self.display_thread.join(timeout=1.0)

            cv2.destroyWindow(self.window_name)

        except Exception as e:
            print(f"Visualization close error: {e}")

    def get_visualization_stats(self) -> Dict[str, float]:
        """Get visualization performance statistics"""
        if not self.render_times:
            return {}

        stats = {
            'avg_render_time_ms': np.mean(self.render_times) * 1000,
            'max_render_time_ms': np.max(self.render_times) * 1000,
            'total_frames': self.frame_count,
            'render_calls': len(self.render_times)
        }

        if self.frame_rates:
            stats.update({
                'avg_fps': np.mean(self.frame_rates),
                'current_fps': self.frame_rates[-1] if self.frame_rates else 0
            })

        return stats

class ComprehensiveDataLogger(IDataLogger):
    """Professional data logging system with multiple output formats"""

    def __init__(self, base_filename: str = "exploration_session",
                 profiler: Optional[PerformanceProfiler] = None):
        self.base_filename = base_filename
        self.profiler = profiler

        # Data storage
        self.sensor_data_log = []
        self.pose_history = []
        self.system_events = []

        # Session information
        self.session_start_time = time.time()
        self.session_id = self._generate_session_id()

        print(f"📊 Data logger initialized: {base_filename}")
        print(f"   Session ID: {self.session_id}")

    def _generate_session_id(self) -> str:
        """Generate unique session identifier"""
        return time.strftime("%Y%m%d_%H%M%S")

    def log_sensor_data(self, sensor_data: Any) -> None:
        """Log sensor data with timestamp"""
        try:
            log_entry = {
                'timestamp': time.time(),
                'sensor_type': str(sensor_data.sensor_type) if hasattr(sensor_data, 'sensor_type') else 'unknown',
                'processing_time': getattr(sensor_data, 'processing_time', 0),
                'data_summary': self._summarize_sensor_data(sensor_data)
            }

            self.sensor_data_log.append(log_entry)

            # Keep only recent data to manage memory
            if len(self.sensor_data_log) > 1000:
                self.sensor_data_log = self.sensor_data_log[-1000:]

        except Exception as e:
            print(f"Sensor data logging error: {e}")

    def _summarize_sensor_data(self, sensor_data: Any) -> Dict[str, Any]:
        """Create summary of sensor data for logging"""
        summary = {}

        if hasattr(sensor_data, 'data') and isinstance(sensor_data.data, dict):
            data = sensor_data.data

            # Count different data types
            if 'points_3d' in data:
                summary['point_count'] = len(data['points_3d'])

            if 'passages' in data:
                summary['passages_count'] = len(data['passages'])

            if 'pillars' in data:
                summary['pillars_count'] = len(data['pillars'])

            if 'front_clearance' in data:
                summary['front_clearance'] = data['front_clearance']

        return summary

    def log_robot_pose(self, pose: RobotPose) -> None:
        """Log robot pose data"""
        try:
            pose_entry = {
                'timestamp': time.time(),
                'x': pose.x,
                'y': pose.y,
                'theta': pose.theta,
                'uncertainty': getattr(pose, 'uncertainty', 0)
            }

            self.pose_history.append(pose_entry)

            # Keep reasonable history size
            if len(self.pose_history) > 5000:
                self.pose_history = self.pose_history[-5000:]

        except Exception as e:
            print(f"Pose logging error: {e}")

    def log_system_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Log system events"""
        try:
            event_entry = {
                'timestamp': time.time(),
                'event_type': event_type,
                'data': event_data
            }

            self.system_events.append(event_entry)

        except Exception as e:
            print(f"Event logging error: {e}")

    def save_exploration_results(self, results: Dict[str, Any]) -> str:
        """Save complete exploration results"""
        try:
            # Add session metadata
            results['session_metadata'] = {
                'session_id': self.session_id,
                'start_time': self.session_start_time,
                'end_time': time.time(),
                'duration': time.time() - self.session_start_time
            }

            # Add logged data
            results['sensor_data_summary'] = self._create_sensor_summary()
            results['pose_statistics'] = self._create_pose_statistics()
            results['system_events'] = self.system_events

            # Save main results file
            filename = f"{self.base_filename}_{self.session_id}.json"

            import json
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            print(f"💾 Exploration results saved: {filename}")

            # Save additional data files
            self._save_pose_trajectory()
            self._save_sensor_statistics()

            return filename

        except Exception as e:
            print(f"Results saving error: {e}")
            return ""

    def _create_sensor_summary(self) -> Dict[str, Any]:
        """Create summary of sensor data"""
        summary = {
            'total_sensor_readings': len(self.sensor_data_log),
            'sensor_types': {},
            'avg_processing_time': 0
        }

        if self.sensor_data_log:
            # Count by sensor type
            for entry in self.sensor_data_log:
                sensor_type = entry['sensor_type']
                if sensor_type not in summary['sensor_types']:
                    summary['sensor_types'][sensor_type] = 0
                summary['sensor_types'][sensor_type] += 1

            # Average processing time
            processing_times = [entry['processing_time'] for entry in self.sensor_data_log if entry['processing_time'] > 0]
            if processing_times:
                summary['avg_processing_time'] = np.mean(processing_times)

        return summary

    def _create_pose_statistics(self) -> Dict[str, Any]:
        """Create pose movement statistics"""
        stats = {
            'total_poses': len(self.pose_history),
            'total_distance': 0,
            'avg_speed': 0,
            'position_bounds': {}
        }

        if len(self.pose_history) > 1:
            # Calculate total distance
            total_distance = 0
            for i in range(1, len(self.pose_history)):
                prev = self.pose_history[i-1]
                curr = self.pose_history[i]

                dx = curr['x'] - prev['x']
                dy = curr['y'] - prev['y']
                distance = math.sqrt(dx**2 + dy**2)
                total_distance += distance

            stats['total_distance'] = total_distance

            # Average speed
            total_time = self.pose_history[-1]['timestamp'] - self.pose_history[0]['timestamp']
            if total_time > 0:
                stats['avg_speed'] = total_distance / total_time

            # Position bounds
            x_positions = [pose['x'] for pose in self.pose_history]
            y_positions = [pose['y'] for pose in self.pose_history]

            stats['position_bounds'] = {
                'x_min': min(x_positions),
                'x_max': max(x_positions),
                'y_min': min(y_positions),
                'y_max': max(y_positions)
            }

        return stats

    def _save_pose_trajectory(self) -> None:
        """Save pose trajectory as CSV"""
        try:
            filename = f"{self.base_filename}_trajectory_{self.session_id}.csv"

            import csv
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'x', 'y', 'theta', 'uncertainty'])

                for pose in self.pose_history:
                    writer.writerow([
                        pose['timestamp'],
                        pose['x'], 
                        pose['y'],
                        pose['theta'],
                        pose.get('uncertainty', 0)
                    ])

            print(f"📊 Trajectory saved: {filename}")

        except Exception as e:
            print(f"Trajectory save error: {e}")

    def _save_sensor_statistics(self) -> None:
        """Save sensor statistics"""
        try:
            filename = f"{self.base_filename}_sensors_{self.session_id}.json"

            sensor_stats = self._create_sensor_summary()

            import json
            with open(filename, 'w') as f:
                json.dump(sensor_stats, f, indent=2)

            print(f"📊 Sensor statistics saved: {filename}")

        except Exception as e:
            print(f"Sensor stats save error: {e}")

    def load_previous_session(self) -> Optional[Dict[str, Any]]:
        """Load previous exploration session"""
        # Implementation would search for previous session files
        # and load the most recent one
        return None

    def get_logging_stats(self) -> Dict[str, Any]:
        """Get data logging statistics"""
        return {
            'session_id': self.session_id,
            'session_duration': time.time() - self.session_start_time,
            'sensor_logs': len(self.sensor_data_log),
            'pose_logs': len(self.pose_history),
            'system_events': len(self.system_events)
        }

if __name__ == "__main__":
    print("📺 Professional Visualization and Monitoring Components Initialized")

    # Test visualizer
    visualizer = RealTimeVisualizer()
    print(f"✅ Real-time visualizer: {visualizer.window_size[0]}x{visualizer.window_size[1]}")
    print(f"✅ Multi-threaded rendering: {visualizer.enable_threading}")

    # Test data logger
    logger = ComprehensiveDataLogger()
    print(f"✅ Data logger: {logger.session_id}")

    print("🚀 Ready for high-performance visualization and monitoring")
    print("⚡ C++ optimization targets: OpenCV rendering and data serialization")
