#!/bin/bash

return_code=0

check()
{
    if [ $? = 0 ]; then
    	echo "$1 OK"
    else
	    echo "$1 ERR"
        return_code=1
    fi
}

RUN_BINARY=${RUN_WITH:-python}

$RUN_BINARY -m unittest test.test_popmaboss
check "popmaboss"
$RUN_BINARY -m unittest test.test_ensemble
check "ensemble"
$RUN_BINARY -m unittest test.test_loadsbml
check "loadsbml"
$RUN_BINARY -m unittest test.test_probtrajs_cmaboss
check "probtrajs_cmaboss"
$RUN_BINARY -m unittest test.test_uppmaboss_cmaboss
check "uppmaboss_cmaboss"
$RUN_BINARY -m unittest test.test_observed_graph_cmaboss
check "observed_graph_cmaboss"

exit $return_code
