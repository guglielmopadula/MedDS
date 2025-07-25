variables_dict={
    "o2":[0,1e+08,"oxygen","bio","bgc"],
    "nppv":[-1e+08,1e+08,"netprimaryprodution","bio","bgc"],
    "dissic":[0,1e+08,"dissolvedinorganiccarbon","car","bgc"],
    "talk":[0,1e+08,"alkalinity","car","bgc"],
    "ph":[0,1e+08,"ph","car","bgc"],
    "spco2":[0,1e+08,"surfacepressureCO2","co2","bgc"],
    "fgco2":[0,1e+08,"surfacefluxCO2","co2","bgc"],
    "nh4":[0,1e+08,"ammonium","nut","bgc"],
    "po4":[0,1e+08,"phosphate","nut","bgc"],
    "no3":[0,1e+08,"nitrate","nut","bgc"],
    "si":[0,1e+08,"silicate","nut","bgc"],
    "kd490":[0,1e+08,"lightattenuationcoefficient","optics","bgc"],
    "chl":[0,1e+08,"chlorophyll","pft","bgc"],
    "diatoChla":[0,1e+08,"diatomschrolophyll","pft","bgc"],
    "dinoChla":[0,1e+08,"dinoflagellateschlorophyll","pft","bgc"],
    'nanoChla':[0,1e+08,"nanophytoplanktonchlorophyll","pft","bgc"],
    'picoChla':[0,1e+08,"picophytoplanktonchlorophyll","pft","bgc"],
    'diatoC':[0,1e+08,"diatomscarbon","pft","bgc"],
    'dinoC':[0,1e+08,"dinoflagellatescarbon","pft","bgc"],
    'nanoC':[0,1e+08,"nanophytoplanktoncarbon","pft","bgc"],
    'phyc':[0,1e+08,"phytoplanktoncarbon","pft","bgc"],
    'picoC':[0,1e+08,"picophytoplanktoncarbon","pft","bgc"],
    'zooc':[0,1e+08,"zooplankton carbon","pft","bgc"],         
    "so":[0,1e+08,"salinity","sal","phy"],
    "zos":[0,1e+08,"sealevel","ssh","phy"],
    "thetao":[0,1e+08,"tempeature","tem","phy"],
    "bottomT":[0,1e+08,"seabedtempeature","tem","phy"],
    "uo":[-1e+08,1e+08,"eastwardvelocity","cur","phy"],
    "vo":[-1e+08,1e+08,"northwardvelocity","vo","phy"],
    "wo":[-1e+08,1e+08,"upwardvelocity","wcur","phy"],
    "mlotst":[0,1e+08,"oceanmixedlayer","mld","phy"],
}

dataset_names="cmems_mod_med_{}-{}_anfc_4.2km_P1D-m"
static_dataset="cmems_mod_med_{}_anfc_4.2km_static"

reanalysis_names={"bgc":"med-ogs-{}-rean-d","phy":"med-cmcc-{}-rean-d"}
