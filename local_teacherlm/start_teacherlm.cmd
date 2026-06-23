@echo off
setlocal
title TeacherLM Launcher
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_teacherlm.ps1" %*
if errorlevel 1 (
  echo.
  echo TeacherLM could not start. Read the error above, then press any key.
  pause >nul
)
endlocal
