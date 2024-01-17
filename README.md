# Energy-Comparator
Small project to help picking electricity and gas contracts.

## Requirements
 - numpy
 - pandas

## Usage
To use this tool:

1. Compute your average gas consumption, power consumption (divided in low-tariff and high-tariff) and power generation (divide in low-tariff and high-tariff).
2. (optional) Compute/estimiate your standard deviation.
3. (optional) Populate a covariance matrix.
4. Obtain (i.e. create/get) a csv file with contracts to compare. The csv file should contain the following fields:
```csv
name,type,gasconstant,gasconstant_unit,gasvariable,powerconstant,powerconstant_unit,powervariable_low,powervariable_high,powergen_low,powergen_high,bonus
```
 
Then, run the program using, e.g:
```bash
python3 calculate.py --profile-gas 1700 --profile-power-low 2700 --profile-power-high 2800 --profile-power-gen-low 3000 --profile-power-gen-high 5000 --profile-covariance-matrix "[ [60,20,20,-7,-14], [20,60,8,0,0], [20,8,60,0,0], [-7,0,0,60,0], [-14,0,0,0,60] ]"
`
