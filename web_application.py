from flask import Flask, render_template, Response, jsonify
import threading
import cv2
import json
import time
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
from video_processing_module import VideoProcessor

app = Flask(__name__)

# Global variables
video_processor = None
processing_thread = None
stop_event = threading.Event()

# Initialize data stores for analytics
productivity_history = []  # List to store productivity metrics over time
zone_history = []  # List to store zone occupancy over time

def background_processing():
    """Background thread for video processing"""
    global video_processor, stop_event
    
    while not stop_event.is_set():
        if video_processor:
            success = video_processor.process_frame()
            if not success:
                break
            
            # Record metrics every 10 seconds for historical analysis
            if int(time.time()) % 10 == 0:
                # Get current metrics
                metrics = video_processor.get_productivity_metrics()
                timestamp = datetime.now()
                
                # Store productivity metrics
                for person_id, person_data in metrics['person_metrics'].items():
                    productivity_history.append({
                        'timestamp': timestamp,
                        'person_id': person_id,
                        'productive_time': person_data['productive_time'],
                        'break_time': person_data['break_time'],
                        'productivity_percentage': person_data['productivity_percentage'],
                        'zone': person_data['current_zone']
                    })
                
                # Store zone occupancy
                for zone_name, zone_data in metrics['zone_metrics'].items():
                    zone_history.append({
                        'timestamp': timestamp,
                        'zone': zone_name,
                        'occupancy': zone_data['current_occupancy']
                    })
            
            # Small delay to reduce CPU usage
            time.sleep(0.01)

def generate_video_feed():
    """Generator function for video streaming route"""
    global video_processor
    
    while True:
        if video_processor and video_processor.current_frame is not None:
            # Encode frame as JPEG
            _, jpeg = cv2.imencode('.jpg', video_processor.current_frame)
            frame_bytes = jpeg.tobytes()
            
            # Yield frame in multipart response format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)

def generate_heatmap_feed():
    """Generator function for heatmap streaming route"""
    global video_processor
    
    while True:
        if video_processor and video_processor.current_frame is not None:
            # Generate heatmap
            heatmap = video_processor.generate_heatmap()
            
            # Encode frame as JPEG
            _, jpeg = cv2.imencode('.jpg', heatmap)
            frame_bytes = jpeg.tobytes()
            
            # Yield frame in multipart response format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route for streaming the processed video feed"""
    return Response(generate_video_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/heatmap_feed')
def heatmap_feed():
    """Route for streaming the occupancy heatmap"""
    return Response(generate_heatmap_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/metrics')
def get_metrics():
    """API endpoint for current productivity metrics"""
    global video_processor
    
    if video_processor:
        metrics = video_processor.get_productivity_metrics()
        anomalies = video_processor.detect_anomalies()
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'anomalies': anomalies
        })
    else:
        return jsonify({'error': 'Video processor not initialized'})

@app.route('/historical_data')
def get_historical_data():
    """API endpoint for historical productivity data"""
    global productivity_history, zone_history
    
    # Convert to dataframes for easier processing
    if productivity_history:
        prod_df = pd.DataFrame(productivity_history)
        zone_df = pd.DataFrame(zone_history)
        
        # Aggregate productivity by hour
        prod_df['hour'] = prod_df['timestamp'].dt.floor('H')
        hourly_prod = prod_df.groupby(['hour', 'person_id'])['productivity_percentage'].mean().reset_index()
        
        # Aggregate zone occupancy by hour
        zone_df['hour'] = zone_df['timestamp'].dt.floor('H')
        hourly_zone = zone_df.groupby(['hour', 'zone'])['occupancy'].mean().reset_index()
        
        return jsonify({
            'productivity': hourly_prod.to_dict(orient='records'),
            'zone_occupancy': hourly_zone.to_dict(orient='records')
        })
    else:
        return jsonify({
            'productivity': [],
            'zone_occupancy': []
        })

@app.route('/productivity_chart')
def productivity_chart():
    """Generate productivity trend chart"""
    global productivity_history
    
    if not productivity_history:
        # Return placeholder image if no data
        return jsonify({'chart': ''})
    
    # Convert to dataframe
    df = pd.DataFrame(productivity_history)
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Group by 10-minute windows and person
    df['time_window'] = df['timestamp'].dt.floor('10min')
    grouped = df.groupby(['time_window', 'person_id'])['productivity_percentage'].mean().reset_index()
    
    # Create line chart
    fig = px.line(
        grouped, 
        x='time_window', 
        y='productivity_percentage', 
        color='person_id',
        title='Productivity Percentage Over Time',
        labels={'time_window': 'Time', 'productivity_percentage': 'Productivity %', 'person_id': 'Person ID'}
    )
    
    # Convert to JSON for frontend
    chart_json = fig.to_json()
    return jsonify({'chart': chart_json})

@app.route('/zone_heatmap')
def zone_heatmap():
    """Generate zone occupancy heatmap chart"""
    global zone_history
    
    if not zone_history:
        # Return placeholder if no data
        return jsonify({'chart': ''})
    
    # Convert to dataframe
    df = pd.DataFrame(zone_history)
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Group by hour and zone
    df['hour'] = df['timestamp'].dt.floor('H')
    pivot = df.pivot_table(
        index='zone', 
        columns='hour',
        values='occupancy',
        aggfunc='mean'
    ).fillna(0)
    
    # Create heatmap
    fig = px.imshow(
        pivot,
        labels=dict(x="Hour", y="Zone", color="Occupancy"),
        title="Zone Occupancy Heatmap",
        color_continuous_scale="Viridis"
    )
    
    # Convert to JSON for frontend
    chart_json = fig.to_json()
    return jsonify({'chart': chart_json})

@app.route('/start', methods=['POST'])
def start_processing():
    """Start video processing with the specified camera"""
    global video_processor, processing_thread, stop_event
    
    # Stop any existing processing
    if processing_thread and processing_thread.is_alive():
        stop_event.set()
        processing_thread.join()
    
    # Reset stop event
    stop_event.clear()
    
    try:
        # Initialize video processor (use camera 0 for demo)
        video_processor = VideoProcessor(camera_id=0)
        
        # Start background processing thread
        processing_thread = threading.Thread(target=background_processing)
        processing_thread.daemon = True
        processing_thread.start()
        
        return jsonify({'status': 'success', 'message': 'Video processing started'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop', methods=['POST'])
def stop_processing():
    """Stop video processing"""
    global processing_thread, stop_event
    
    if processing_thread and processing_thread.is_alive():
        stop_event.set()
        processing_thread.join()
        return jsonify({'status': 'success', 'message': 'Video processing stopped'})
    else:
        return jsonify({'status': 'error', 'message': 'No active processing to stop'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
