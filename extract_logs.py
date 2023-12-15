import re
import os
from pathlib import Path

def parse_logfile(logfile):
    n_match = re.search(r"Running with N = (\d+)", logfile)
    time_match = re.search(r"\[(.+?)<.+\]", logfile)  
    time_string = time_match.group(1)
    time_parts = time_string.split(":")
    seconds = int(time_parts[-1])
    if len(time_parts) == 2:
        minutes = int(time_parts[0])
        seconds += 60 * minutes
    elif len(time_parts) == 3:
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        seconds += 60 * (minutes + 60 * hours)
    return int(n_match.group(1)), seconds

def parse_and_save_log_data(log_dir, output_file):
    data_list = []
    for logfile in Path(log_dir).glob("*.txt"):
        with open(logfile, "r") as f:
            logfile_content = f.read()
            n, time_in_seconds = parse_logfile(logfile_content)
            data_list.append({"N": n, "Time (seconds)": time_in_seconds})

    df = pd.DataFrame(data_list)
    df.to_csv(output_file, index=False)
