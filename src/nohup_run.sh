#!/bin/bash

nohup python3 pool_executor.py > my.log 2>&1 &
echo $! > save_pid.txt