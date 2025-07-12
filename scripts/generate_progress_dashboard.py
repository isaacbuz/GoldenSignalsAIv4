#!/usr/bin/env python3
"""
Generate a visual progress dashboard for the project
Creates an HTML dashboard with charts and metrics
"""
import json
from pathlib import Path
from datetime import datetime

def generate_dashboard():
    """Generate an HTML dashboard with progress visualization"""
    
    # Load progress data
    progress_file = Path("progress_data.json")
    if progress_file.exists():
        with open(progress_file) as f:
            data = json.load(f)
    else:
        print("No progress data found. Run 'python scripts/progress_tracker.py report' first.")
        return
    
    # Calculate metrics
    total_tasks = sum(len(m["tasks"]) for m in data["milestones"].values())
    completed_tasks = sum(
        1 for m in data["milestones"].values()
        for t in m["tasks"].values()
        if t["progress"] == 100
    )
    overall_progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GoldenSignalsAI Progress Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            color: #666;
            margin-top: 5px;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .milestone {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }}
        .milestone-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .progress-bar {{
            background: #e0e0e0;
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
        }}
        .progress-fill {{
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            transition: width 0.3s ease;
        }}
        .status-badge {{
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .status-completed {{ background: #4caf50; color: white; }}
        .status-in_progress {{ background: #ff9800; color: white; }}
        .status-planned {{ background: #9e9e9e; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>GoldenSignalsAI Progress Dashboard</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{overall_progress:.1f}%</div>
                <div class="metric-label">Overall Progress</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{completed_tasks}/{total_tasks}</div>
                <div class="metric-label">Tasks Completed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data['metrics']['docs_consolidated']}/{data['metrics']['docs_total']}</div>
                <div class="metric-label">Docs Consolidated</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data['metrics']['components_created']}/{data['metrics']['components_planned']}</div>
                <div class="metric-label">Components Created</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>Progress by Milestone</h2>
            <canvas id="progressChart" width="400" height="200"></canvas>
        </div>
        
        <div class="chart-container">
            <h2>Task Status Distribution</h2>
            <canvas id="statusChart" width="400" height="200"></canvas>
        </div>
        
        <h2>Milestone Details</h2>
"""
    
    # Add milestone details
    for milestone_id, milestone in data["milestones"].items():
        tasks = milestone.get("tasks", {})
        if tasks:
            progress = sum(t["progress"] for t in tasks.values()) / len(tasks)
        else:
            progress = 0
        
        html += f"""
        <div class="milestone">
            <div class="milestone-header">
                <h3>{milestone['name']}</h3>
                <span class="status-badge status-{milestone['status']}">{milestone['status'].replace('_', ' ').title()}</span>
            </div>
            <p>Period: {milestone['start_date']} to {milestone['end_date']}</p>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress}%"></div>
            </div>
            <p>{progress:.1f}% Complete</p>
        </div>
"""
    
    # Add chart data
    milestone_names = [m["name"] for m in data["milestones"].values()]
    milestone_progress = []
    for m in data["milestones"].values():
        tasks = m.get("tasks", {})
        if tasks:
            progress = sum(t["progress"] for t in tasks.values()) / len(tasks)
        else:
            progress = 0
        milestone_progress.append(progress)
    
    # Count task statuses
    status_counts = {"done": 0, "in_progress": 0, "planned": 0, "blocked": 0}
    for m in data["milestones"].values():
        for t in m.get("tasks", {}).values():
            status = t.get("status", "planned")
            if status in status_counts:
                status_counts[status] += 1
    
    html += f"""
    </div>
    
    <script>
        // Progress by Milestone Chart
        const progressCtx = document.getElementById('progressChart').getContext('2d');
        new Chart(progressCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(milestone_names)},
                datasets: [{{
                    label: 'Progress %',
                    data: {json.dumps(milestone_progress)},
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100
                    }}
                }}
            }}
        }});
        
        // Task Status Chart
        const statusCtx = document.getElementById('statusChart').getContext('2d');
        new Chart(statusCtx, {{
            type: 'doughnut',
            data: {{
                labels: ['Completed', 'In Progress', 'Planned', 'Blocked'],
                datasets: [{{
                    data: [{status_counts['done']}, {status_counts['in_progress']}, {status_counts['planned']}, {status_counts['blocked']}],
                    backgroundColor: [
                        'rgba(76, 175, 80, 0.8)',
                        'rgba(255, 152, 0, 0.8)',
                        'rgba(158, 158, 158, 0.8)',
                        'rgba(244, 67, 54, 0.8)'
                    ]
                }}]
            }}
        }});
    </script>
</body>
</html>
"""
    
    # Save dashboard
    dashboard_path = Path("progress_dashboard.html")
    with open(dashboard_path, 'w') as f:
        f.write(html)
    
    print(f"✅ Dashboard generated: {dashboard_path.absolute()}")
    print(f"📊 Overall Progress: {overall_progress:.1f}%")
    print(f"📈 Tasks Completed: {completed_tasks}/{total_tasks}")

if __name__ == "__main__":
    generate_dashboard() 