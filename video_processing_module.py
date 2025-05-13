import cv2
import numpy as np
import torch
from ultralytics import YOLO
from bytetrack.byte_tracker import BYTETracker
import time
from datetime import datetime
import json

class VideoProcessor:
    def __init__(self, camera_id=0, model_path="yolov8n.pt", zone_config_path="zones.json"):
        """
        Initialize the video processor with camera feed and object detection model
        
        Args:
            camera_id: Camera identifier (int for webcam, string for IP camera URL)
            model_path: Path to YOLOv8 model weights
            zone_config_path: Path to zone configuration JSON file
        """
        # Initialize camera
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open camera {camera_id}")
        
        # Load YOLOv8 model
        self.model = YOLO(model_path)
        
        # Initialize ByteTrack tracker
        self.tracker = BYTETracker(
            track_thresh=0.5,  # Detection confidence threshold
            track_buffer=30,   # Frames to keep track of lost objects
            match_thresh=0.8,  # IoU threshold for matching
            frame_rate=30      # Expected frame rate
        )
        
        # Load zone configurations
        self.zones = self._load_zones(zone_config_path)
        
        # Initialize person tracking data
        self.person_locations = {}  # person_id -> current_zone
        self.person_timestamps = {}  # person_id -> {zone: [enter_time, exit_time]}
        
        # Current frame and processing time tracking
        self.current_frame = None
        self.fps = 0
        self.last_time = time.time()
        
    def _load_zones(self, config_path):
        """Load zone definitions from config file"""
        with open(config_path, 'r') as f:
            zone_config = json.load(f)
        
        zones = {}
        for zone in zone_config:
            # Each zone has a name, type (desk, meeting, break), and polygon coordinates
            zones[zone['name']] = {
                'type': zone['type'],
                'polygon': np.array(zone['coordinates'], dtype=np.int32),
                'occupancy': 0,
                'productive': zone['type'] in ['desk', 'meeting']  # Desks and meeting rooms are productive
            }
        return zones
    
    def point_in_zone(self, point, zone_polygon):
        """Check if a point is inside a polygon zone"""
        return cv2.pointPolygon(zone_polygon, point) > 0
    
    def get_person_zone(self, bbox):
        """Determine which zone a person is in based on their bounding box"""
        # Use the center bottom point of bbox as the person's position (their feet)
        x1, y1, x2, y2 = bbox
        person_point = (int((x1 + x2) / 2), int(y2))
        
        for zone_name, zone_data in self.zones.items():
            if self.point_in_zone(person_point, zone_data['polygon']):
                return zone_name
        
        return "untracked"  # Person is not in any defined zone
    
    def update_person_location(self, person_id, zone_name, timestamp):
        """Update a person's location and time tracking"""
        prev_zone = self.person_locations.get(person_id, None)
        
        # If this is a new person or they've changed zones
        if prev_zone != zone_name:
            # Record exit from previous zone if applicable
            if prev_zone:
                if person_id not in self.person_timestamps:
                    self.person_timestamps[person_id] = {}
                if prev_zone not in self.person_timestamps[person_id]:
                    self.person_timestamps[person_id][prev_zone] = []
                    
                # Add exit time to the last enter time for this zone
                zone_times = self.person_timestamps[person_id][prev_zone]
                if zone_times and len(zone_times) % 2 == 1:  # There's an enter without an exit
                    zone_times.append(timestamp)
                    
                # Update zone occupancy count
                if prev_zone in self.zones:
                    self.zones[prev_zone]['occupancy'] = max(0, self.zones[prev_zone]['occupancy'] - 1)
            
            # Record entry to new zone
            if zone_name != "untracked":
                if person_id not in self.person_timestamps:
                    self.person_timestamps[person_id] = {}
                if zone_name not in self.person_timestamps[person_id]:
                    self.person_timestamps[person_id][zone_name] = []
                    
                self.person_timestamps[person_id][zone_name].append(timestamp)
                
                # Update zone occupancy count
                if zone_name in self.zones:
                    self.zones[zone_name]['occupancy'] += 1
            
            # Update current location
            self.person_locations[person_id] = zone_name
    
    def process_frame(self):
        """Process a single frame from the video feed"""
        # Read frame from camera
        success, frame = self.cap.read()
        if not success:
            return False
        
        self.current_frame = frame.copy()
        timestamp = datetime.now()
        
        # Run object detection
        results = self.model(frame, classes=0)  # class 0 is person in COCO dataset
        
        # Extract detections in ByteTrack format
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                detections.append([x1, y1, x2, y2, confidence])
        
        # Run tracker
        if detections:
            track_results = self.tracker.update(
                np.array(detections),
                [frame.shape[0], frame.shape[1]],
                [frame.shape[0], frame.shape[1]]
            )
            
            # Process each tracked person
            for track in track_results:
                person_id = track.track_id
                bbox = track.tlbr  # top-left bottom-right bounding box
                
                # Find which zone the person is in
                zone_name = self.get_person_zone(bbox)
                
                # Update person location and time tracking
                self.update_person_location(person_id, zone_name, timestamp)
                
                # Visualize bounding box and ID
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(self.current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(self.current_frame, f"ID: {person_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw zones on the frame
        for zone_name, zone_data in self.zones.items():
            color = (0, 255, 0) if zone_data['productive'] else (0, 0, 255)
            cv2.polylines(self.current_frame, [zone_data['polygon']], True, color, 2)
            centroid = np.mean(zone_data['polygon'], axis=0).astype(int)
            cv2.putText(self.current_frame, f"{zone_name} ({zone_data['occupancy']})", 
                       tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Calculate FPS
        current_time = time.time()
        self.fps = 1 / (current_time - self.last_time)
        self.last_time = current_time
        
        # Display FPS on the frame
        cv2.putText(self.current_frame, f"FPS: {self.fps:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return True
    
    def get_productivity_metrics(self):
        """Calculate productivity metrics for all tracked individuals"""
        metrics = {
            'person_metrics': {},
            'zone_metrics': {zone: {'total_occupancy_time': 0, 'current_occupancy': data['occupancy']} 
                            for zone, data in self.zones.items()}
        }
        
        current_time = datetime.now()
        
        # Calculate metrics for each person
        for person_id, zones in self.person_timestamps.items():
            productive_time = 0
            break_time = 0
            
            for zone_name, timestamps in zones.items():
                zone_type = self.zones.get(zone_name, {}).get('type', 'unknown')
                is_productive = zone_type in ['desk', 'meeting']
                
                # Calculate time spent in this zone
                zone_time = 0
                for i in range(0, len(timestamps), 2):
                    start_time = timestamps[i]
                    # If there's no exit time, use current time
                    end_time = timestamps[i+1] if i+1 < len(timestamps) else current_time
                    duration = (end_time - start_time).total_seconds()
                    zone_time += duration
                    
                    # Add to zone metrics
                    metrics['zone_metrics'][zone_name]['total_occupancy_time'] += duration
                
                # Add to person's productive or break time
                if is_productive:
                    productive_time += zone_time
                else:
                    break_time += zone_time
            
            # Calculate productivity percentage
            total_time = productive_time + break_time
            productivity_percentage = (productive_time / total_time * 100) if total_time > 0 else 0
            
            # Store person metrics
            metrics['person_metrics'][person_id] = {
                'productive_time': productive_time,
                'break_time': break_time,
                'total_time': total_time,
                'productivity_percentage': productivity_percentage,
                'current_zone': self.person_locations.get(person_id, 'unknown')
            }
        
        return metrics
    
    def detect_anomalies(self):
        """Detect abnormal patterns or rule violations"""
        anomalies = []
        current_time = datetime.now()
        
        # Check for idle time anomalies
        idle_threshold = 30 * 60  # 30 minutes in seconds
        
        for person_id, current_zone in self.person_locations.items():
            if current_zone in self.zones and self.zones[current_zone]['productive']:
                # Check last activity time
                timestamps = self.person_timestamps.get(person_id, {}).get(current_zone, [])
                if timestamps and (current_time - timestamps[-1]).total_seconds() > idle_threshold:
                    anomalies.append({
                        'type': 'idle_time',
                        'person_id': person_id,
                        'zone': current_zone,
                        'idle_duration': (current_time - timestamps[-1]).total_seconds()
                    })
        
        # Check for overcapacity meeting rooms
        for zone_name, zone_data in self.zones.items():
            if zone_data['type'] == 'meeting' and zone_data.get('capacity'):
                if zone_data['occupancy'] > zone_data['capacity']:
                    anomalies.append({
                        'type': 'overcapacity',
                        'zone': zone_name,
                        'current_occupancy': zone_data['occupancy'],
                        'capacity': zone_data['capacity']
                    })
        
        return anomalies
    
    def generate_heatmap(self):
        """Generate a heatmap of workspace occupancy"""
        # Create a blank heatmap image
        heatmap = np.zeros(self.current_frame.shape[:2], dtype=np.uint8)
        
        # Set intensity values based on zone occupancy
        for zone_name, zone_data in self.zones.items():
            # Create a mask for the zone
            mask = np.zeros(heatmap.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [zone_data['polygon']], 255)
            
            # Set intensity based on occupancy (normalized)
            intensity = min(255, zone_data['occupancy'] * 50)  # Scale factor can be adjusted
            heatmap[mask > 0] = intensity
        
        # Apply colormap
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Blend with original frame
        alpha = 0.5
        blended = cv2.addWeighted(self.current_frame, 1 - alpha, heatmap, alpha, 0)
        
        return blended
    
    def run(self):
        """Main processing loop"""
        while True:
            if not self.process_frame():
                break
            
            # Display the processed frame
            cv2.imshow("Workspace Monitor", self.current_frame)
            
            # Generate and display heatmap
            heatmap = self.generate_heatmap()
            cv2.imshow("Occupancy Heatmap", heatmap)
            
            # Calculate and print metrics every 5 seconds
            if int(time.time()) % 5 == 0:
                metrics = self.get_productivity_metrics()
                anomalies = self.detect_anomalies()
                print(f"Metrics: {metrics}")
                if anomalies:
                    print(f"Anomalies detected: {anomalies}")
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

# Example zone configuration (would normally be loaded from JSON file)
example_zones = [
    {
        "name": "Desk 1",
        "type": "desk",
        "coordinates": [[100, 200], [300, 200], [300, 400], [100, 400]]
    },
    {
        "name": "Meeting Room A",
        "type": "meeting",
        "capacity": 6,
        "coordinates": [[400, 100], [700, 100], [700, 300], [400, 300]]
    },
    {
        "name": "Break Area",
        "type": "break",
        "coordinates": [[100, 500], [350, 500], [350, 700], [100, 700]]
    }
]

# For testing/demo purposes
if __name__ == "__main__":
    # Save example zone configuration
    with open("zones.json", "w") as f:
        json.dump(example_zones, f)
    
    # Initialize and run video processor
    processor = VideoProcessor(camera_id=0)  # Use default webcam
    processor.run()
