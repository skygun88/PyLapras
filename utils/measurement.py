import os
import psutil


def check_memory_usage():
    process = psutil.Process(os.getpid())
    usage_gb = process.memory_info().rss / 2**30 
    return usage_gb