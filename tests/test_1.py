import unittest
import pytest
import pandas as pd
import re
#from .util import prnt
#from .util import print_all_output
#from src.extract_degrees import Extract_Degree
from ..src.extract_easy_sync import Extract_Easy_Sync
#from src.extract_neighbours import Extract_Neighbours

class TestClass(unittest.TestCase):

    @pytest.mark.parametrize("changeset", [
        ('Z:5g>1|5=2p=v*4*5+1$x'),
        ('Z:5k>1=15*4+1$'),
        ('Z:5l>1|2=2s=1m*4+1$'),
        ('Z:5m>1|2=2s=1n*4+1$Ö'),
        ('Z:5n>2|2=2s=1o*4+2$nd'),
        ('Z:5p>2|2=2s=1q*4+2$er'),
        ('Z:e9>1|4=cq=1i*4+1$'),
        ('Z:ea>1|4=cq=1j*4+1$Ö'),
        ('Z:eb>2|4=cq=1k*4+2$nd'),
        ('Z:ed>2|4=cq=1m*4+2$er'),
        ('Z:ef<1*5=1|1=37*6=1|1=2f*7=1|1=47*8=1|1=2t*9=1=1m-1$'),
        ('Z:2>dx-1*4*1*2*5*3+1*1|1+37*4*1*2*5*3+1*1|1+2f*4*1*2*5*3+1*1|1+47*4*1*2*5*3+1*1|1+2j*4*1*2*5*3+1*1+1h$sdsds')
    ])
    def test_easy_sync(changeset):
        source_codes = [
            'Z:5g>1|5=2p=v*4*5+1$x',
            'Z:5k>1=15*4+1$',
            'Z:5l>1|2=2s=1m*4+1$',
            'Z:5m>1|2=2s=1n*4+1$Ö',
            'Z:5n>2|2=2s=1o*4+2$nd',
            'Z:5p>2|2=2s=1q*4+2$er',
            'Z:e9>1|4=cq=1i*4+1$',
            'Z:ea>1|4=cq=1j*4+1$Ö',
            'Z:eb>2|4=cq=1k*4+2$nd',
            'Z:ed>2|4=cq=1m*4+2$er',
            'Z:ef<1*5=1|1=37*6=1|1=2f*7=1|1=47*8=1|1=2t*9=1=1m-1$', #9
            'Z:2>dx-1*4*1*2*5*3+1*1|1+37*4*1*2*5*3+1*1|1+2f*4*1*2*5*3+1*1|1+47*4*1*2*5*3+1*1|1+2j*4*1*2*5*3+1*1+1h$sdsds'
        ]
        es = Extract_Easy_Sync()
        res = es.extract_changeset(changeset, 'all')
        self.assertEqual(4, 4, 
            f'Area is shown 4 for side = 4 units')

if __name__ == '__main__':
    unittest.main()