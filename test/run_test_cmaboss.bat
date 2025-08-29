echo off

if ($env:RUN_WITH) { $RUN_BINARY = $env:RUN_WITH } else { $RUN_BINARY = "python" };
set /A FAIL=0

$RUN_BINARY -m unittest test.test_popmaboss.TestPopPMaBoSS.test_fork
call:check
$RUN_BINARY -m unittest test.test_popmaboss.TestPopPMaBoSS.test_log_growth
call:check

$RUN_BINARY -m unittest test.test_ensemble
call:check

$RUN_BINARY -m unittest test.test_loadsbml
call:check
    
$RUN_BINARY -m unittest test.test_probtrajs_cmaboss
call:check

$RUN_BINARY -m unittest test.test_uppmaboss_cmaboss
call:check

$RUN_BINARY -m unittest test.test_observed_graph_cmaboss
call:check

$RUN_BINARY -m unittest test.test_cellcycle
call:check

exit /b %FAIL%



:check 
if %ERRORLEVEL% NEQ 0 (
    set /A FAIL=1
)
exit /b 