// Dashboard JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const currentMetricsEl = document.getElementById('currentMetrics');
    const anomaliesEl = document.getElementById('anomalies');
    const productivityChartEl = document.getElementById('productivityChart');
    const zoneHeatmapEl = document.getElementById('zoneHeatmap');
    
    // Initialize Plotly charts
    let productivityChart = null;
    let zoneHeatmap = null;
    
    // Initialize data refresh intervals
    let metricsInterval = null;
    let chartsInterval = null;
    
    // Start/Stop button event listeners
    startBtn.addEventListener('click', startProcessing);
    stopBtn.addEventListener('click', stopProcessing);
    
    // Initial UI state
    stopBtn.disabled = true;
    
    // Functions
    function startProcessing() {
        fetch('/start', {
            method: 'POST',
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Update UI
                startBtn.disabled = true;
                stopBtn.disabled = false;
                
                // Start data refresh intervals
                startDataRefresh();
                
                // Show success message
                showAlert('success', 'Video processing started successfully');
            } else {
                showAlert('danger', `Error: ${data.message}`);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showAlert('danger', 'Failed to start video processing');
        });
    }
    
    function stopProcessing() {
        fetch('/stop', {
            method: 'POST',
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Update UI
                startBtn.disabled = false;
                stopBtn.disabled = true;
                
                // Stop data refresh intervals
                stopDataRefresh();
                
                // Show success message
                showAlert('success', 'Video processing stopped');
            } else {
                showAlert('danger', `Error: ${data.message}`);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showAlert('danger', 'Failed to stop video processing');
        });
    }
    
    function startDataRefresh() {
        // Refresh current metrics every 3 seconds
        metricsInterval = setInterval(fetchCurrentMetrics, 3000);
        
        // Refresh charts every 30 seconds
        chartsInterval = setInterval(fetchCharts, 30000);
        
        // Initial data fetch
        fetchCurrentMetrics();
        fetchCharts();
    }
    
    function stopDataRefresh() {
        clearInterval(metricsInterval);
        clearInterval(chartsInterval);
    }
    
    function fetchCurrentMetrics() {
        fetch('/metrics')
        .then(response => response.json())
        .then(data => {
            updateMetricsDisplay(data);
            updateAnomaliesDisplay(data.anomalies);
        })
        .catch(error => {
            console.error('Error fetching metrics:', error);
        });
    }
    
    function fetchCharts() {
        // Fetch productivity chart data
        fetch('/productivity_chart')
        .then(response => response.json())
        .then(data => {
            if (data.chart) {
                updateProductivityChart(data.chart);
            }
        })
        .catch(error => {
            console.error('Error fetching productivity chart:', error);
        });
        
        // Fetch zone heatmap data
        fetch('/zone_heatmap')
        .then(response => response.json())
        .then(data => {
            if (data.chart) {
                updateZoneHeatmap(data.chart);
            }
        })
        .catch(error => {
            console.error('Error fetching zone heatmap:', error);
        });
    }
    
    function updateMetricsDisplay(data) {
        if (!data || !data.metrics) {
            currentMetricsEl.innerHTML = '<p class="text-muted">No metrics available</p>';
            return;
        }
        
        let html = '';
        
        // Zone metrics
        html += '<div class="mb-4"><h5>Zone Occupancy</h5>';
        html += '<div class="list-group">';
        
        const zones = data.metrics.zone_metrics;
        for (const [zoneName, zoneData] of Object.entries(zones)) {
            html += `
                <div class="zone-item">
                    <span class="zone-name">${zoneName}</span>
                    <span class="zone-occupancy">${zoneData.current_occupancy} people</span>
                </div>
            `;
        }
        
        html += '</div></div>';
        
        // Person metrics
        html += '<div><h5>Person Productivity</h5>';
        
        const persons = data.metrics.person_metrics;
        for (const [personId, personData] of Object.entries(persons)) {
            const productivityPercentage = Math.round(personData.productivity_percentage);
            const productiveTime = formatTime(personData.productive_time);
            const breakTime = formatTime(personData.break_time);
            
            html += `
                <div class="person-metrics">
                    <div class="person-id">Person ID: ${personId}</div>
                    <div class="productivity-bar">
                        <div class="productivity-fill" style="width: ${productivityPercentage}%"></div>
                    </div>
                    <div class="d-flex justify-content-between">
                        <small>Productivity: ${productivityPercentage}%</small>
                        <small>Current Zone: ${personData.current_zone}</small>
                    </div>
                    <div class="mt-2 small">
                        <div>Productive Time: ${productiveTime}</div>
                        <div>Break Time: ${breakTime}</div>
                    </div>
                </div>
            `;
        }
        
        html += '</div>';
        
        currentMetricsEl.innerHTML = html;
    }
    
    function updateAnomaliesDisplay(anomalies) {
        if (!anomalies || anomalies.length === 0) {
            anomaliesEl.innerHTML = '<p class="text-muted">No anomalies detected</p>';
            return;
        }
        
        let html = '';
        
        anomalies.forEach(anomaly => {
            let details = '';
            
            if (anomaly.type === 'idle_time') {
                const idleMinutes = Math.round(anomaly.idle_duration / 60);
                details = `Person ${anomaly.person_id} has been idle in ${anomaly.zone} for ${idleMinutes} minutes`;
            } else if (anomaly.type === 'overcapacity') {
                details = `${anomaly.zone} is overcapacity (${anomaly.current_occupancy}/${anomaly.capacity})`;
            }
            
            html += `
                <div class="anomaly-item">
                    <div class="anomaly-title">${capitalizeFirstLetter(anomaly.type)} Detected</div>
                    <div class="anomaly-details">${details}</div>
                </div>
            `;
        });
        
        anomaliesEl.innerHTML = html;
    }
    
    function updateProductivityChart(chartData) {
        try {
            const chartConfig = JSON.parse(chartData);
            Plotly.react(productivityChartEl, chartConfig.data, chartConfig.layout);
        } catch (error) {
            console.error('Error updating productivity chart:', error);
        }
    }
    
    function updateZoneHeatmap(chartData) {
        try {
            const chartConfig = JSON.parse(chartData);
            Plotly.react(zoneHeatmapEl, chartConfig.data, chartConfig.layout);
        } catch (error) {
            console.error('Error updating zone heatmap:', error);
        }
    }
    
    // Helper functions
    function formatTime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        
        if (hours > 0) {
            return `${hours}h ${minutes}m`;
        } else {
            return `${minutes}m`;
        }
    }
    
    function capitalizeFirstLetter(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
    
    function showAlert(type, message) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed top-0 end-0 m-3`;
        alertDiv.setAttribute('role', 'alert');
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        document.body.appendChild(alertDiv);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            alertDiv.classList.remove('show');
            setTimeout(() => alertDiv.remove(), 300);
        }, 5000);
    }
});
