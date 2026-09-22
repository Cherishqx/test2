@echo off
REM ============================================================
REM  RAG.py 自动化测试一键运行脚本（Windows）
REM  双击本文件或在命令行执行：run_tests.bat
REM ============================================================
cd /d %~dp0
echo [INFO] 工作目录: %CD%
echo [INFO] 开始运行 RAG.py 全部 30 条测试用例...
echo.
python -m pytest tests -v
set EXITCODE=%ERRORLEVEL%
echo.
if %EXITCODE%==0 (
    echo [DONE] 全部用例通过。
) else (
    echo [DONE] 存在失败/跳过的用例，退出码: %EXITCODE%
    echo        失败用例即为缺陷线索，详见上方详细输出。
)
pause
