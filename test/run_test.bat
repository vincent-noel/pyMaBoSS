echo off

if ($env:RUN_WITH) { $RUN_BINARY = $env:RUN_WITH } else { $RUN_BINARY = "python" };
set /A FAIL=0

$RUN_BINARY -m unittest test.test_conversion
call:check

$RUN_BINARY -m unittest test.test_loadmodels
call:check

$RUN_BINARY -m unittest test.test_observed_graph
call:check

$RUN_BINARY -m unittest test.test_pipelines
call:check

$RUN_BINARY -m unittest test.test_probtrajs
call:check

$RUN_BINARY -m unittest test.test_statdist
call:check

$RUN_BINARY -m unittest test.test_types
call:check

$RUN_BINARY -m unittest test.test_uppmaboss
call:check


exit /b %FAIL%



:check 
if %ERRORLEVEL% NEQ 0 (
    set /A FAIL=1
)
exit /b 