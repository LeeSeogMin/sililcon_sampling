@echo off
echo START > experiment_output.log
echo Starting Clova Experiment... >> experiment_output.log
python code\06_clova_experiment.py --personas outputs\personas\personas_100_seed43.json --out-dir results\clova_experiment_seed43 --resume >> experiment_output.log 2>&1
echo Finished with exit code %errorlevel% >> experiment_output.log
echo END >> experiment_output.log
