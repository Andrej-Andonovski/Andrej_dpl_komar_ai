@echo off
"C:\Users\Andrej\AppData\Local\Programs\Python\Python313\python.exe" -u "C:\Users\Andrej\Desktop\fpl ai\pipeline\watch_new_signings.py" >> "C:\Users\Andrej\Desktop\fpl ai\data\intel\signing_watch\cron_log.txt" 2>&1
echo BATCH_EXIT_CODE=%ERRORLEVEL% >> "C:\Users\Andrej\Desktop\fpl ai\data\intel\signing_watch\cron_log.txt"
