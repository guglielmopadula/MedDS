basins_dict={}
eps=1e-5

#[min_longitude,max_longitude,min_latitude,max_latitude]
basins_dict["alb"]=tuple([-5.5-eps,-1-eps,32-eps,40+eps])
basins_dict["swm1"]=tuple([-1-eps,5-eps,32-eps,39.5+eps])
basins_dict["swm2"]=tuple([-5-eps,9.25-eps,32-eps,39.5+eps])
basins_dict["ion1"]=tuple([9.25-eps,15-eps,32-eps,36.75+eps])
basins_dict["ion2"]=tuple([15-eps,21.85+eps,30-eps,36.75+eps])
basins_dict["adr1"]=tuple([11.5-eps,20+eps,42.5+eps,46+eps])
basins_dict["lev3"]=tuple([26.25-eps,33+eps,30-eps,33.60+eps])
basins_dict["lev4"]=tuple([33+eps,37+eps, 30-eps,38+eps])
basins_dict["adr2a"]=tuple([17.95-eps,20+eps,40+eps,42.5+eps])
basins_dict["adr2b"]=tuple([15-eps,17.95-eps,40.5+eps,42+eps])
basins_dict["adr2c"]=tuple([12-eps,17.95-eps,42+eps,42.5+eps])
basins_dict["tyr1a"]=tuple([9.25-eps,11.5-eps,42.5+eps,46+eps])
basins_dict["tyr1b"]=tuple([9.25-eps,12-eps,42+eps,42.5+eps])
basins_dict["tyr1c"]=tuple([9.25-eps,15-eps,41.25+eps,42+eps])
basins_dict["ion3a"]=tuple([15-eps,21.85-eps,36.75+eps,38+eps])
basins_dict["ion3b"]=tuple([15.5-eps,21.85-eps,38+eps,38.2+eps])
basins_dict["ion3c"]=tuple([16.23-eps,21.85-eps,38.2+eps,40+eps])
basins_dict["ion3d"]=tuple([16.23-eps,17.95-eps,40-eps,40.5+eps])
basins_dict["tyr2a"]=tuple([9.25-eps,15-eps,36.75+eps,41.25+eps])
basins_dict["tyr2b"]=tuple([15-eps,15.5-eps,38+eps,40.5+eps])
basins_dict["tyr2c"]=tuple([15.5-eps,16.23-eps,38.2+eps,40.5+eps])
basins_dict["lev2a"]=tuple([28+eps,33+eps,33.60+eps,38+eps])
basins_dict["nwm"]=tuple([-1-eps,9.25-eps,39.5+eps,46+eps])
basins_dict["lev2b"]=tuple([26.25-eps,28+eps,33.60+eps,35.28+eps])
basins_dict["lev1a"]=tuple([21.85-eps,24-eps,30-eps,35.28+eps])
basins_dict["lev1b"]=tuple([24-eps,26.25-eps,30-eps,35.10+eps])
basins_dict["aega"]=tuple([21.85-eps,24-eps,35.28+eps,42+eps])
basins_dict["aegb"]=tuple([24-eps,26.25-eps,35.10+eps,42+eps])
basins_dict["aegc"]=tuple([26.25-eps,28+eps,35.28+eps,42+eps])
basins_dict["nadr"]=tuple([11.5-eps,20+eps,42.5+eps,46+eps])


#[min_bound,max_bound,name,dataset_subname,dataset_index]

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