from netCDF4 import Dataset,num2date
import numpy as np

grid_path = '/data/project6/kesf/ROMS/L2SCB_AP/share/'
grid_name = 'roms_grd.nc'

grid_nc = Dataset(grid_path+grid_name,'r')
lat_nc = np.array(grid_nc.variables['lat_rho'][:,:])
lon_nc = np.array(grid_nc.variables['lon_rho'][:,:])
h_nc = np.array(grid_nc.variables['h'][:,:])
mask_nc = np.array(grid_nc.variables['mask_rho'][:,:])
pm_nc = np.array(grid_nc.variables['pm'])
pn_nc = np.array(grid_nc.variables['pn'])

