@echo off
echo Debug Test Starting > debug_output.txt
dir >> debug_output.txt
python -c "print('Python Hello')" > python_test.txt 2>&1
echo Python Exit Code: %errorlevel% >> debug_output.txt
