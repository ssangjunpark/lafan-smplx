from datetime import datetime

start_time = datetime.strptime("2026-01-06 21:04:38", "%Y-%m-%d %H:%M:%S")
end_time = datetime.strptime("2026-01-10 19:50:30", "%Y-%m-%d %H:%M:%S")

time_difference = end_time - start_time

days = time_difference.days
seconds = time_difference.seconds
hours = seconds // 3600
minutes = (seconds % 3600) // 60
secs = seconds % 60

print(f"Time difference: {days} days, {hours} hours, {minutes} minutes, {secs} seconds")
