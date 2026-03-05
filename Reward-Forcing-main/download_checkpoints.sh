#!/bin/bash

echo "======================================================"
echo "Starting model download..."
echo "======================================================"

echo "Downloading Wan2.1-T2V-1.3B..."
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir checkpoints/Wan2.1-T2V-1.3B 

echo "Downloading Wan2.1-T2V-14B..."
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir checkpoints/Wan2.1-T2V-14B 

echo "Downloading VideoReward..."
huggingface-cli download KlingTeam/VideoReward --local-dir checkpoints/Videoreward 

echo "Downloading ODE Initialization..."
huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir .

echo "Downloading Reward Forcing..."
huggingface-cli download JaydenLu666/Reward-Forcing-T2V-1.3B --local-dir checkpoints/Reward-Forcing-T2V-1.3B

echo "======================================================"
echo "Finished downloading models!"
ls -R checkpoints
echo "======================================================"