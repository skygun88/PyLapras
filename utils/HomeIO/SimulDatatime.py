from datetime import datetime

def sdt_to_dt(sdt):
    date, day_split, time = str(sdt).split()
    year, month, day = [int(x) for x in date.split('-')]
    hour, minute, second = [int(x) for x in time.split(':')]
    if day_split == '오전':
        if hour == 12:
            hour = 0
    else:
        if hour != 12:
            hour += 12
    
    dt = datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
    return dt