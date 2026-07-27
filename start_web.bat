@echo off
setlocal

rem Fixed ports for the malchan development web application.
set "BACKEND_HOST=127.0.0.1"
set "BACKEND_PORT=8001"
set "FRONTEND_HOST=127.0.0.1"
set "FRONTEND_PORT=5174"

rem The React application uses this absolute API URL instead of the default Vite proxy.
set "VITE_API_BASE=http://%BACKEND_HOST%:%BACKEND_PORT%/api"
set "MALCHAN_CORS_ORIGINS=http://%FRONTEND_HOST%:%FRONTEND_PORT%,http://localhost:%FRONTEND_PORT%"

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    echo Activate the malchan virtual environment before running this file.
    pause
    exit /b 1
)

where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm was not found on PATH.
    echo Install Node.js and make sure npm is available.
    pause
    exit /b 1
)

echo Starting malchan backend at http://%BACKEND_HOST%:%BACKEND_PORT% ...
start "malchan backend" /D "%~dp0" cmd /k python -m uvicorn "malchan.app:create_app" --factory --reload --host %BACKEND_HOST% --port %BACKEND_PORT%

echo Starting malchan frontend at http://%FRONTEND_HOST%:%FRONTEND_PORT% ...
start "malchan frontend" /D "%~dp0frontend" cmd /k npm run dev -- --host %FRONTEND_HOST% --port %FRONTEND_PORT% --strictPort

echo.
echo malchan startup commands were launched.
echo Frontend: http://%FRONTEND_HOST%:%FRONTEND_PORT%
echo Backend : http://%BACKEND_HOST%:%BACKEND_PORT%

endlocal
