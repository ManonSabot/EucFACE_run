df              = pd.DataFrame(f.variables['Rainf'][:,0], columns=['Rainf']) # 'Rainfall+snowfall'
df['Evap']      = f.variables['Evap'][:,0] # 'Total evaporation'
df['Qs']        = f.variables['Qs'][:,0]   # 'Surface runoff'
df['Qsb']       = f.variables['Qsb'][:,0]  # 'Subsurface runoff'
df['Qrecharge'] = f.variables['Qrecharge'][:,0]
df['SoilMoist'] = f.variables['SoilMoist'][:,0,0]
df['ECanop']    = f.variables['ECanop'][:,0,0] # 'Wet canopy evaporation'
df['ESoil']     = f.variables['ESoil'][:,0,0]  # 'evaporation from soil'
df['TVeg']       = f.variables['TVeg'][:,0,0]  # 'Vegetation transpiration'



# df['SnowDepth'] = f.variables['SnowDepth'][:,0]*1000.
# df['CanopInt']  = f.variables['CanopInt'][:,0]
