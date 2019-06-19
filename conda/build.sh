$PYTHON setup.py install --single-version-externally-managed --record=record.txt
mkdir -p ${PREFIX}/bin
cp scripts/UpPMaBoSS.py ${PREFIX}/bin
cp scripts/MBSS_FormatTable.py ${PREFIX}/bin