
#
# class Result example using maboss communication layer: to be merged with pyMaBoSS/maboss/result.py
#

from __future__ import print_function
import maboss.maboss_server.comm

class Result:

    def __init__(self, mbcli, simulation, hints):
        if "check" in hints and hints["check"]:
            command = maboss.maboss_server.comm.CHECK_COMMAND
        else:
            command = maboss.maboss_server.comm.RUN_COMMAND

        client_data = maboss.maboss_server.comm.ClientData(simulation.getNetwork(), simulation.getConfig(), command)

        data = maboss.maboss_server.comm.DataStreamer.buildStreamData(client_data, hints)
        #print("sending data", data)

        data = mbcli.send(data)
        #print("received data", data)

        self._result_data = maboss.maboss_server.comm.DataStreamer.parseStreamData(data, hints )

    def getResultData(self):
        return self._result_data
