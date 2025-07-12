#!/usr/bin/env python3
"""
Automated progress tracking for GoldenSignalsAI milestones
Tracks tasks, generates reports, and provides status updates
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import argparse

class MilestoneTracker:
    def __init__(self, data_file: str = "progress_data.json"):
        self.data_file = data_file
        self.load_data()
    
    def load_data(self):
        """Load existing progress data or initialize new data"""
        if Path(self.data_file).exists():
            with open(self.data_file) as f:
                self.data = json.load(f)
        else:
            self.data = self.initialize_data()
            self.save_data()
    
    def initialize_data(self) -> Dict:
        """Initialize the milestone tracking data structure"""
        return {
            "project": "GoldenSignalsAI",
            "start_date": "2024-06-30",
            "milestones": {
                "week_1_2": {
                    "name": "Foundation Phase",
                    "status": "completed",
                    "start_date": "2024-06-30",
                    "end_date": "2024-07-07",
                    "tasks": {
                        "docs_inventory": {
                            "name": "Documentation Inventory",
                            "status": "done",
                            "progress": 100,
                            "assignee": "team",
                            "completed": "2024-06-30"
                        },
                        "ui_logging": {
                            "name": "Frontend Logging System",
                            "status": "done",
                            "progress": 100,
                            "assignee": "team",
                            "completed": "2024-06-30"
                        },
                        "storybook_setup": {
                            "name": "Storybook Configuration",
                            "status": "done",
                            "progress": 100,
                            "assignee": "team",
                            "completed": "2024-06-30"
                        },
                        "env_config": {
                            "name": "Environment Configuration",
                            "status": "done",
                            "progress": 100,
                            "assignee": "team",
                            "completed": "2024-06-30"
                        },
                        "debug_panel": {
                            "name": "Debug Panel Component",
                            "status": "done",
                            "progress": 100,
                            "assignee": "team",
                            "completed": "2024-06-30"
                        }
                    }
                },
                "week_3_4": {
                    "name": "Core Infrastructure",
                    "status": "in_progress",
                    "start_date": "2024-07-08",
                    "end_date": "2024-07-21",
                    "tasks": {
                        "core_docs": {
                            "name": "Core Documentation",
                            "status": "in_progress",
                            "progress": 30,
                            "assignee": "docs_team",
                            "subtasks": {
                                "project_overview": {"done": True},
                                "architecture": {"done": True},
                                "setup_guide": {"done": True},
                                "api_reference": {"done": False}
                            }
                        },
                        "component_library": {
                            "name": "Component Library",
                            "status": "planned",
                            "progress": 0,
                            "assignee": "ui_team"
                        },
                        "test_infrastructure": {
                            "name": "Test Infrastructure",
                            "status": "planned",
                            "progress": 0,
                            "assignee": "ui_team"
                        },
                        "mock_data": {
                            "name": "Mock Data System",
                            "status": "planned",
                            "progress": 0,
                            "assignee": "ui_team"
                        }
                    }
                },
                "week_5_6": {
                    "name": "Module Development",
                    "status": "planned",
                    "start_date": "2024-07-22",
                    "end_date": "2024-08-04",
                    "tasks": {}
                },
                "week_7_8": {
                    "name": "Testing & Quality",
                    "status": "planned",
                    "start_date": "2024-08-05",
                    "end_date": "2024-08-18",
                    "tasks": {}
                },
                "week_9_10": {
                    "name": "Performance & Operations",
                    "status": "planned",
                    "start_date": "2024-08-19",
                    "end_date": "2024-09-01",
                    "tasks": {}
                },
                "week_11_12": {
                    "name": "Migration & Polish",
                    "status": "planned",
                    "start_date": "2024-09-02",
                    "end_date": "2024-09-15",
                    "tasks": {}
                }
            },
            "metrics": {
                "docs_consolidated": 12,
                "docs_total": 95,
                "components_created": 1,
                "components_planned": 50,
                "test_coverage": 0,
                "bugs_fixed": 0,
                "story_points_completed": 35,
                "velocity": 35
            },
            "team": {
                "docs_team": ["alice", "bob"],
                "ui_team": ["charlie", "dave", "eve"],
                "devops": ["frank"]
            }
        }
    
    def update_task(self, milestone: str, task: str, status: str, progress: int):
        """Update a specific task's status and progress"""
        if milestone in self.data["milestones"] and task in self.data["milestones"][milestone]["tasks"]:
            self.data["milestones"][milestone]["tasks"][task].update({
                "status": status,
                "progress": progress,
                "updated": datetime.now().isoformat()
            })
            
            # Update milestone status based on tasks
            self._update_milestone_status(milestone)
            self.save_data()
            print(f"✅ Updated {task} in {milestone}: {status} ({progress}%)")
        else:
            print(f"❌ Task {task} not found in milestone {milestone}")
    
    def _update_milestone_status(self, milestone: str):
        """Update milestone status based on task completion"""
        tasks = self.data["milestones"][milestone]["tasks"]
        if not tasks:
            return
        
        total_progress = sum(task["progress"] for task in tasks.values())
        avg_progress = total_progress / len(tasks)
        
        if avg_progress == 100:
            self.data["milestones"][milestone]["status"] = "completed"
        elif avg_progress > 0:
            self.data["milestones"][milestone]["status"] = "in_progress"
        else:
            self.data["milestones"][milestone]["status"] = "planned"
    
    def add_task(self, milestone: str, task_id: str, task_name: str, assignee: str = "team"):
        """Add a new task to a milestone"""
        if milestone in self.data["milestones"]:
            self.data["milestones"][milestone]["tasks"][task_id] = {
                "name": task_name,
                "status": "planned",
                "progress": 0,
                "assignee": assignee,
                "created": datetime.now().isoformat()
            }
            self.save_data()
            print(f"✅ Added task {task_id} to {milestone}")
        else:
            print(f"❌ Milestone {milestone} not found")
    
    def generate_report(self, format: str = "markdown") -> str:
        """Generate a progress report in markdown or json format"""
        if format == "json":
            return json.dumps(self.data, indent=2)
        
        report = ["# GoldenSignalsAI Progress Report\n"]
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"**Project Start:** {self.data['start_date']}\n")
        
        # Overall Progress
        total_tasks = sum(len(m["tasks"]) for m in self.data["milestones"].values())
        completed_tasks = sum(
            1 for m in self.data["milestones"].values()
            for t in m["tasks"].values()
            if t["progress"] == 100
        )
        overall_progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        report.append("## Overall Progress")
        report.append(f"- **Tasks Completed:** {completed_tasks}/{total_tasks} ({overall_progress:.1f}%)")
        report.append(f"- **Documentation:** {self.data['metrics']['docs_consolidated']}/{self.data['metrics']['docs_total']} files consolidated")
        report.append(f"- **Components:** {self.data['metrics']['components_created']}/{self.data['metrics']['components_planned']} created")
        report.append(f"- **Velocity:** {self.data['metrics']['velocity']} story points/sprint\n")
        
        # Milestone Details
        report.append("## Milestones")
        
        for milestone_id, milestone in self.data["milestones"].items():
            status_emoji = {
                "completed": "✅",
                "in_progress": "🚧",
                "planned": "📋",
                "blocked": "🚫"
            }.get(milestone["status"], "❓")
            
            report.append(f"\n### {status_emoji} {milestone['name']} ({milestone_id})")
            report.append(f"**Status:** {milestone['status'].replace('_', ' ').title()}")
            report.append(f"**Period:** {milestone['start_date']} to {milestone['end_date']}")
            
            if milestone["tasks"]:
                report.append("\n**Tasks:**")
                
                total_progress = 0
                for task_id, task in milestone["tasks"].items():
                    task_emoji = {
                        "done": "✅",
                        "in_progress": "🚧",
                        "planned": "📋",
                        "blocked": "🚫"
                    }.get(task["status"], "❓")
                    
                    report.append(f"- {task_emoji} **{task['name']}** ({task_id})")
                    report.append(f"  - Progress: {task['progress']}%")
                    report.append(f"  - Assignee: {task['assignee']}")
                    
                    if "subtasks" in task:
                        report.append("  - Subtasks:")
                        for subtask, done in task["subtasks"].items():
                            check = "✅" if done else "⬜"
                            report.append(f"    - {check} {subtask}")
                    
                    total_progress += task["progress"]
                
                avg_progress = total_progress / len(milestone["tasks"])
                report.append(f"\n**Milestone Progress:** {avg_progress:.1f}%")
        
        # Metrics
        report.append("\n## Key Metrics")
        for metric, value in self.data["metrics"].items():
            metric_name = metric.replace('_', ' ').title()
            report.append(f"- **{metric_name}:** {value}")
        
        # Team
        report.append("\n## Team Assignment")
        for team, members in self.data["team"].items():
            team_name = team.replace('_', ' ').title()
            report.append(f"- **{team_name}:** {', '.join(members)}")
        
        # Next Actions
        report.append("\n## Next Actions")
        for milestone in self.data["milestones"].values():
            if milestone["status"] == "in_progress":
                for task_id, task in milestone["tasks"].items():
                    if task["status"] != "done":
                        report.append(f"- Continue {task['name']} ({task['assignee']})")
        
        return "\n".join(report)
    
    def save_data(self):
        """Save the current data to file"""
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def get_current_sprint(self) -> Optional[str]:
        """Get the current sprint based on dates"""
        today = datetime.now().date()
        for milestone_id, milestone in self.data["milestones"].items():
            start = datetime.strptime(milestone["start_date"], "%Y-%m-%d").date()
            end = datetime.strptime(milestone["end_date"], "%Y-%m-%d").date()
            if start <= today <= end:
                return milestone_id
        return None
    
    def get_burndown_data(self) -> Dict:
        """Get data for burndown chart"""
        burndown = {
            "planned": [],
            "actual": [],
            "dates": []
        }
        
        total_points = sum(len(m["tasks"]) * 8 for m in self.data["milestones"].values())
        remaining = total_points
        
        for milestone in self.data["milestones"].values():
            burndown["dates"].append(milestone["end_date"])
            planned_burn = len(milestone["tasks"]) * 8
            burndown["planned"].append(remaining)
            
            actual_burn = sum(t["progress"] / 100 * 8 for t in milestone["tasks"].values())
            remaining -= actual_burn
            burndown["actual"].append(remaining)
            
            remaining -= planned_burn
        
        return burndown

def main():
    parser = argparse.ArgumentParser(description="GoldenSignalsAI Progress Tracker")
    parser.add_argument("action", choices=["report", "update", "add", "burndown"],
                       help="Action to perform")
    parser.add_argument("--milestone", help="Milestone ID (e.g., week_3_4)")
    parser.add_argument("--task", help="Task ID")
    parser.add_argument("--status", choices=["planned", "in_progress", "done", "blocked"],
                       help="Task status")
    parser.add_argument("--progress", type=int, help="Progress percentage (0-100)")
    parser.add_argument("--name", help="Task name")
    parser.add_argument("--assignee", help="Task assignee")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown",
                       help="Report format")
    
    args = parser.parse_args()
    tracker = MilestoneTracker()
    
    if args.action == "report":
        print(tracker.generate_report(args.format))
    
    elif args.action == "update":
        if not all([args.milestone, args.task, args.status, args.progress is not None]):
            print("Error: --milestone, --task, --status, and --progress required for update")
            return
        tracker.update_task(args.milestone, args.task, args.status, args.progress)
    
    elif args.action == "add":
        if not all([args.milestone, args.task, args.name]):
            print("Error: --milestone, --task, and --name required for add")
            return
        tracker.add_task(args.milestone, args.task, args.name, 
                        args.assignee or "team")
    
    elif args.action == "burndown":
        data = tracker.get_burndown_data()
        print("Burndown Chart Data:")
        print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main() 