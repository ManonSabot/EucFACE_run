#!/bin/bash

cd /g/data/w35/mm3972/cable/EucFACE/EucFACE_run
source activate science

cp /g/data/w35/mm3972/cable/EucFACE/EucFACE_met/hyds10/*.nc ./met/
python3 ./run_cable_site.py
mv ./outputs/*.nc ./outputs/hyds10

cp /g/data/w35/mm3972/cable/EucFACE/EucFACE_met/bch1.5/*.nc ./met/
python3 ./run_cable_site.py
mv ./outputs/*.nc ./outputs/bch1.5

cp /g/data/w35/mm3972/cable/EucFACE/EucFACE_met/sfc2/*.nc ./met/
python3 ./run_cable_site.py
mv ./outputs/*.nc ./outputs/sfc2

cp /g/data/w35/mm3972/cable/EucFACE/EucFACE_met/ssat2/*.nc ./met/
python3 ./run_cable_site.py
mv ./outputs/*.nc ./outputs/ssat2

cp /g/data/w35/mm3972/cable/EucFACE/EucFACE_met/cnsd2/*.nc ./met/
python3 ./run_cable_site.py
mv ./outputs/*.nc ./outputs/cnsd2

cp /g/data/w35/mm3972/cable/EucFACE/EucFACE_met/sucs2/*.nc ./met/
python3 ./run_cable_site.py
mv ./outputs/*.nc ./outputs/sucs2

cp /g/data/w35/mm3972/cable/EucFACE/EucFACE_met/css1.5/*.nc ./met/
python3 ./run_cable_site.py
mv ./outputs/*.nc ./outputs/css1.5

cp /g/data/w35/mm3972/cable/EucFACE/EucFACE_met/swilt2/*.nc ./met/
python3 ./run_cable_site.py
mv ./outputs/*.nc ./outputs/swilt2
