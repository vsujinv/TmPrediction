import openpyxl
import subprocess as sub
import sys


source_file = sys.argv[1]
workbook = openpyxl.load_workbook(source_file)
sheet = workbook['Worksheet']



def reading():
    """ retrieve the SMILES from the .xlsx """
    global smiles_collector
    smiles_collector = []
    for cell in sheet['A']:
        smiles_collector.append(str('-:"') + str(cell.value) + str('" '))

def writing():
    """ provide a container .sdf """
    reading()
    i = 0
    for smile in smiles_collector:
        conversion = str("obabel {0} -osdf > output_{1}.sdf --gen2d".format(smile, i))
        sub.call(conversion, shell=True)
        i += 1


reading()
writing()
sys.exit()

