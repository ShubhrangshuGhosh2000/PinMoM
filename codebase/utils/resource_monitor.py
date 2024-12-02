import os
import psutil
import time
import torch
import subprocess

# Define the ResourceMonitor class
class ResourceMonitor():
    def __init__(self, cuda_index=None, verbose_mon=None):
        self.cuda_index = cuda_index
        self.verbose_mon = verbose_mon
        
        # Get the initial RAM, already occupied at the start of the simulation by other processes.
        self.initial_ram_usage = self.__get_initial_ram_usage()  # in bytes

        # Set nvidia command for GPU monitoring
        self.gpu_monitor_cmd = ["nvidia-smi", "--id=" + str(self.cuda_index), "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"]
        # Get the initial GPU memory which would be already occupied at the start of the simulation by other processes.
        self.initial_gpu_usage = self.__get_initial_gpu_usage()  # in MB

        # initialize a few counters to keep track of the peak resource usage
        self.peak_ram_usage = self.initial_ram_usage
        self.peak_gpu_usage = self.initial_gpu_usage


    def __get_initial_ram_usage(self):
        ram_usage_process = psutil.Process(os.getpid())
        initial_ram_usage = ram_usage_process.memory_info().rss  # in bytes
        return initial_ram_usage


    def __get_initial_gpu_usage(self):
        initial_gpu_usage = 0
        process = subprocess.Popen(self.gpu_monitor_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, _ = process.communicate()
        lines = output.decode().split('\n')
        if len(lines) > 1:
            total_memory, used_memory = map(int, lines[0].split(','))
            initial_gpu_usage = used_memory  # in MB
        return initial_gpu_usage


    def monitor_peak_ram_usage(self):
        process = psutil.Process(os.getpid())
        crnt_ram_usage = process.memory_info().rss
        if(crnt_ram_usage > self.peak_ram_usage):
            self.peak_ram_usage = crnt_ram_usage
        if(self.verbose_mon): print(f'\n @@@@@@@@ peak_ram_usage: {self.peak_ram_usage} Bytes\n')


    def monitor_peak_gpu_memory_usage(self):
        process = subprocess.Popen(self.gpu_monitor_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, _ = process.communicate()
        lines = output.decode().split('\n')
        if len(lines) > 1:
            total_memory, crnt_used_gpu_memory = map(int, lines[0].split(','))
        if(crnt_used_gpu_memory > self.peak_gpu_usage):
            self.peak_gpu_usage = crnt_used_gpu_memory
        if(self.verbose_mon):  print(f"\n @@@@@@@@ GPU core {self.cuda_index}: Total memory: {total_memory} MB, peak_gpu_usage: {self.peak_gpu_usage} MB\n")
        