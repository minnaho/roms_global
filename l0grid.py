from netCDF4 import Dataset,num2date
import numpy as np

grid_path = '/data/project6/kesf/ROMS/USW4/organization/'
grid_name = 'roms_grd.nc'

grid_nc = Dataset(grid_path+grid_name,'r')
lat_nc = np.array(grid_nc.variables['lat_rho'][:,:])
lon_nc = np.array(grid_nc.variables['lon_rho'][:,:])
h_nc = np.array(grid_nc.variables['h'][:,:])
mask_nc = np.array(grid_nc.variables['mask_rho'][:,:])

