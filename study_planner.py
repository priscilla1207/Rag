from datetime import datetime, timedelta

def create_study_plan(topic: str, deadline: str) -> str:
    try:
        due_date = datetime.strptime(deadline, "%Y-%m-%d")
        today = datetime.today()
        days_left = (due_date - today).days

        if days_left <= 0:
            return f"⚠ Deadline {deadline} has already passed or is today."

        plan = f"📅 Study Plan for '{topic}' (Deadline: {deadline})\n"
        plan += "------------------------------------------\n"
        for i in range(1, days_left + 1):
            day = today + timedelta(days=i)
            plan += f"{day.strftime('%Y-%m-%d')}: Study part {i} of {topic}\n"

        plan += "✅ Final day: Revision and practice tests.\n"
        return plan

    except ValueError:
        return "❌ Invalid date format. Use YYYY-MM-DD."
