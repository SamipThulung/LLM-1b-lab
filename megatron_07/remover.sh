#!/bin/bash
# This is a comment: The command below deletes a specific target
pkill -9 -f "multiprocessing.spawn"
rm -rf megatron_07/checkpoints_l
rm -rf megatron_07/checkpoints_m
rm -rf megatron_07/checkpoints_s
rm -rf megatron_07/logs
