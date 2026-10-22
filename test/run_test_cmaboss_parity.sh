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

for module in loadmodels conversion istates probtrajs statdist observed_graph pipelines uppmaboss loadsbml tabularqual; do
    $RUN_BINARY -m unittest test.cmaboss.test_$module
    check "cmaboss/$module"
done

exit $return_code
