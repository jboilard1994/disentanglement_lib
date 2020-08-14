# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:28:11 2020

@author: Jonathan Boilard
"""

def manage_processes(processes, queue, max_process=4):
    """ @author: jboilard 
    from a list of already set processes, manage processes
    starts processes up to a certain maximum number of processes
    terminates the processes once they are over
    
    processes : already set processes from the function get_processes(...)
    results_dir : folder in which the process returns are saved
    queue : element in which the process outputs are saved
    """
    active_processes = []
    return_dicts = []
    while len(processes) > 0 or len(active_processes) > 0:
        # fill active processes list
        for process in processes:
            if process.is_alive() == False and process.exitcode == None and len(active_processes) < 4: #process not started yet
                active_processes.append(process)
                active_processes[-1].start()
                          
        # check if any active_processes has ended
        ended_processes_idx = []
        for i, process in enumerate(active_processes):
            if process.is_alive() == False and process.exitcode == 0: #process has ended
                print("No Error! {}".format(str(process.name)))
                process.terminate()
                processes.remove(process)
                ended_processes_idx.append(i)
                
                #dump
                return_dicts.append(queue.get())
                
                
            elif process.is_alive() == False and process.exitcode == 1: #process has ended
                print(str(process.name) + " ended with an error code : " + str(process.exitcode))
                process.terminate()
                processes.remove(process)
                ended_processes_idx.append(i)
         
        new_active_processes = []
        for i in range(len(active_processes)):
            if not i in ended_processes_idx:
                new_active_processes.append(active_processes[i])
        active_processes = new_active_processes
        
        
        
    return return_dicts
        
        
        