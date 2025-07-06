# deadline_utils.py

import re
from datetime import datetime
from ics import Calendar, Event

def extract_homework_deadlines(text):
    # Regex to find task with deadline pattern
    pattern = r"(?:submit|complete|finish|turn in|deadline for)\s+(.*?)(?:\s+by\s+|\s+on\s+)(\w+\s+\d{1,2},\s+\d{4})(?:\s+at\s+(\d{1,2}:\d{2}\s*(?:AM|PM)))?"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)

    tasks = []
    for task, date_str, time_str in matches:
        try:
            dt_str = date_str
            if time_str:
                dt_str += f" {time_str}"
            dt = datetime.strptime(dt_str.strip(), "%B %d, %Y %I:%M %p") if time_str else datetime.strptime(dt_str.strip(), "%B %d, %Y")
            tasks.append({
                "task": task.strip().capitalize(),
                "due": dt
            })
        except Exception as e:
            print(f"⚠️ Error parsing datetime: {e}")
    return tasks

def create_ics_file(tasks, filename="homework.ics"):
    calendar = Calendar()
    for task in tasks:
        event = Event()
        event.name = task["task"]
        event.begin = task["due"]
        calendar.events.add(event)

    with open(filename, "w") as f:
        f.writelines(calendar)
