cd C:\ml_fraudulent

set "VIRTUAL_ENV=C:\ml_fraudulent\venv"

if defined _OLD_VIRTUAL_PROMPT (
    set "PROMPT=%_OLD_VIRTUAL_PROMPT%"
) else (
    if not defined PROMPT (
        set "PROMPT=$P$G"
    )
    if not defined VIRTUAL_ENV_DISABLE_PROMPT (
        set "_OLD_VIRTUAL_PROMPT=%PROMPT%"
    )
)
if not defined VIRTUAL_ENV_DISABLE_PROMPT (
    if "" NEQ "" (
        set "PROMPT=() %PROMPT%"
    ) else (
        for %%d in ("%VIRTUAL_ENV%") do set "PROMPT=(%%~nxd) %PROMPT%"
    )
)

REM Don't use () to avoid problems with them in %PATH%
if defined _OLD_VIRTUAL_PYTHONHOME goto ENDIFVHOME
    set "_OLD_VIRTUAL_PYTHONHOME=%PYTHONHOME%"
:ENDIFVHOME

set PYTHONHOME=

REM if defined _OLD_VIRTUAL_PATH (
if not defined _OLD_VIRTUAL_PATH goto ENDIFVPATH1
    set "PATH=%_OLD_VIRTUAL_PATH%"
:ENDIFVPATH1
REM ) else (
if defined _OLD_VIRTUAL_PATH goto ENDIFVPATH2
    set "_OLD_VIRTUAL_PATH=%PATH%"
:ENDIFVPATH2

set "PATH=%VIRTUAL_ENV%\Scripts;%PATH%"

@ECHO OFF
ECHO =====================================================
ECHO Script run successfully kindly wait for a while .....
ECHO =====================================================

python manage.py runserver

start chrome http://127.0.0.1:8000/

PAUSE