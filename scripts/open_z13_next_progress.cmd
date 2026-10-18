@echo off
start "" powershell.exe -NoExit -ExecutionPolicy Bypass -File "E:\Dissertation\scripts\z13_next_progress.ps1" -Run "E:\Dissertation\runs\z13_parallel_next_20261018\utility" -Title "FlyVis utility train26"
start "" powershell.exe -NoExit -ExecutionPolicy Bypass -File "E:\Dissertation\scripts\z13_next_progress.ps1" -Run "E:\Dissertation\runs\z13_parallel_next_20261018\differentiable" -Title "FlyVis differentiable readout"
