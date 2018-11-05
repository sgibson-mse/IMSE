import pyuda
import idlbridge as idl

client = pyuda.Client()

shotnumber = 24409

psi_data = client.get('EFM_PSI(R,Z)', shotnumber)

print(psi_data)


