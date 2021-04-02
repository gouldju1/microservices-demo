#Required Packages
import sys
from memory_profiler import profile

#System Path
sys.path.append("../../")

#NER Model
from src import compare

#Sample Text
sample = """
SRT010G900 overlap with 0305900SRT0807E00 overlap with 0305900SRT0706Z00 \
overlap with 0305900SRT0807E00 overlap with 010G900SRT0706Z00 overlap with \
010G900steam cleaned engine added dye and ran truck at high idle found gear \
cover leaking removed hood and bumper drained coolant recovered Freon removed \
coolant reservoir, ps reservoir, both radiator support, upper and lower rad hoses, \
radiator, ac compressor and bracket, alternator, fan, fan shroud, fan hub, removed \
and resealed gear cover reinstalled all removed parts refilled coolant and Freon ran \
truck at high idle no leaks repair completeOIL LEAK EXTERNALUPPER GEAR COVER GASKETLEAKS \
EPR Part Number:430716600 OIL1045962 THURSDAY 31OCT2019 05:00:47 AM
"""

#Approach B
@profile
def run_b():
    return compare.approach_b_pipeline(sample)

if __name__ == '__main__':
    run_b()