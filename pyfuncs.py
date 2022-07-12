import numpy as np

def calc_ij(nc_grd,lat_sites,lon_sites):

    lon_nc = nc_grd.variables['lon_rho'][:,:]
    lat_nc = nc_grd.variables['lat_rho'][:,:]

    nsites = len(lat_sites)
    isites = np.ones(nsites)*np.nan
    jsites = np.ones(nsites)*np.nan

    for s in range(nsites):
        ##################################
        # FIND SITE IN GRIDPOINTS
        ####################################
        min_1D = np.abs( (lat_nc - lat_sites[s])**2 + (lon_nc - lon_sites[s])**2)
        y_site, x_site = np.unravel_index(min_1D.argmin(), min_1D.shape)
        isites[s] = x_site
        jsites[s] = y_site

    return isites, jsites

