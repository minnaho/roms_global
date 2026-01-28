import numpy as np
from netCDF4 import Dataset,num2date

def calc_ij(nc_grd,lat_sites,lon_sites):
    # find closest eta and xi points in grid
    # to given latitude/longitude

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

def match2Darr(arr1,arr2):
    # find x and y points that 
    # match in both 2D arrays
    arr1set = set([tuple(x) for x in arr1.T])
    arr2set = set([tuple(x) for x in arr2.T])
    arrloc = np.array([x for x in arr1set & arr2set])
    return arrloc

def rho_uv(roms_file):
    '''
    calculate u_rho and v_rho from 
    u and v in roms file

    roms_file --> roms output file name (string)
    '''
    # get u and v roms values
    out_nc = Dataset(roms_file,'r')

    # u interpolation
    [s_rho,Mp,L] = np.squeeze(out_nc.variables['u']).shape
    Lp = L+1
    Lm = L-1
    u_temp = 0.5*(np.squeeze(out_nc.variables['u'])[:,:,1:L]+np.squeeze(out_nc.variables['u'])[:,:,:Lm])
    u_rho = np.zeros((s_rho,Mp,Lp))
    u_rho[:,:,1:-1] = u_temp
    u_rho[:,:,0] = u_temp[:,:,0]
    u_rho[:,:,-1] = u_temp[:,:,-1]

    # v interpolation
    [s_rho,M,Lp] = np.squeeze(out_nc.variables['v']).shape
    Mp = M+1
    Mm = M-1
    v_temp = 0.5*(np.squeeze(out_nc.variables['v'])[:,1:M,:]+np.squeeze(out_nc.variables['v'])[:,:Mm,:])
    v_rho = np.zeros((s_rho,Mp,Lp))
    v_rho[:,1:-1,:] = v_temp
    v_rho[:,0,:] = v_temp[:,0,:]
    v_rho[:,-1,:] = v_temp[:,-1,:]


    return u_rho,v_rho

def rho_uv_tind(roms_file,tind):
    '''
    calculate u_rho and v_rho from 
    u and v in roms file

    roms_file --> roms output file name (string)
    tind --> time index
    '''
    # get u and v roms values
    out_nc = Dataset(roms_file,'r')

    # u interpolation
    [s_rho,Mp,L] = np.squeeze(out_nc.variables['u'][tind]).shape
    Lp = L+1
    Lm = L-1
    u_temp = 0.5*(np.squeeze(out_nc.variables['u'])[tind,:,:,1:L]+np.squeeze(out_nc.variables['u'])[tind,:,:,:Lm])
    u_rho = np.zeros((s_rho,Mp,Lp))
    u_rho[:,:,1:-1] = u_temp
    u_rho[:,:,0] = u_temp[:,:,0]
    u_rho[:,:,-1] = u_temp[:,:,-1]

    # v interpolation
    [s_rho,M,Lp] = np.squeeze(out_nc.variables['v'][tind]).shape
    Mp = M+1
    Mm = M-1
    v_temp = 0.5*(np.squeeze(out_nc.variables['v'])[tind,:,1:M,:]+np.squeeze(out_nc.variables['v'])[tind,:,:Mm,:])
    v_rho = np.zeros((s_rho,Mp,Lp))
    v_rho[:,1:-1,:] = v_temp
    v_rho[:,0,:] = v_temp[:,0,:]
    v_rho[:,-1,:] = v_temp[:,-1,:]


    return u_rho,v_rho

def rho_uv_tind_srho(roms_file,tind,srho):
    '''
    calculate u_rho and v_rho from 
    u and v in roms file

    roms_file --> roms output file name (string)
    '''
    # get u and v roms values
    out_nc = Dataset(roms_file,'r')

    # u interpolation
    [Mp,L] = np.squeeze(out_nc.variables['u'][tind,srho]).shape
    Lp = L+1
    Lm = L-1
    u_temp = 0.5*(np.squeeze(out_nc.variables['u'])[tind,srho,:,1:L]+np.squeeze(out_nc.variables['u'])[tind,srho,:,:Lm])
    u_rho = np.zeros((Mp,Lp))
    u_rho[:,1:-1] = u_temp
    u_rho[:,0] = u_temp[:,0]
    u_rho[:,-1] = u_temp[:,-1]

    # v interpolation
    [M,Lp] = np.squeeze(out_nc.variables['v'][tind,srho]).shape
    Mp = M+1
    Mm = M-1
    v_temp = 0.5*(np.squeeze(out_nc.variables['v'])[tind,srho,1:M,:]+np.squeeze(out_nc.variables['v'])[tind,srho,:Mm,:])
    v_rho = np.zeros((Mp,Lp))
    v_rho[1:-1,:] = v_temp
    v_rho[0,:] = v_temp[0,:]
    v_rho[-1,:] = v_temp[-1,:]


    return u_rho,v_rho

def rho_uv_surf(roms_file):
    '''
    calculate u_rho and v_rho from 
    u and v in roms file
    u_surf and v_surf with multiple time indices

    roms_file --> roms output file name (string)
    '''
    # get u and v roms values
    out_nc = Dataset(roms_file,'r')

    # u interpolation
    [s_rho,Mp,L] = np.squeeze(out_nc.variables['u_surf']).shape
    Lp = L+1
    Lm = L-1
    u_temp = 0.5*(np.squeeze(out_nc.variables['u_surf'])[:,:,1:L]+np.squeeze(out_nc.variables['u_surf'])[:,:,:Lm])
    u_rho = np.zeros((s_rho,Mp,Lp))
    u_rho[:,:,1:-1] = u_temp
    u_rho[:,:,0] = u_temp[:,:,0]
    u_rho[:,:,-1] = u_temp[:,:,-1]

    # v interpolation
    [s_rho,M,Lp] = np.squeeze(out_nc.variables['v_surf']).shape
    Mp = M+1
    Mm = M-1
    v_temp = 0.5*(np.squeeze(out_nc.variables['v_surf'])[:,1:M,:]+np.squeeze(out_nc.variables['v_surf'])[:,:Mm,:])
    v_rho = np.zeros((s_rho,Mp,Lp))
    v_rho[:,1:-1,:] = v_temp
    v_rho[:,0,:] = v_temp[:,0,:]
    v_rho[:,-1,:] = v_temp[:,-1,:]


    return u_rho,v_rho

def rho_uv_surf_2d(roms_file):
    '''
    calculate u_rho and v_rho from 
    u and v in roms file
    u_surf and v_surf with 1 time index

    roms_file --> roms output file name (string)
    '''
    # get u and v roms values
    out_nc = Dataset(roms_file,'r')

    # u interpolation
    [Mp,L] = np.squeeze(out_nc.variables['u_surf']).shape
    Lp = L+1
    Lm = L-1
    u_temp = 0.5*(np.squeeze(out_nc.variables['u_surf'])[:,1:L]+np.squeeze(out_nc.variables['u_surf'])[:,:Lm])
    u_rho = np.zeros((Mp,Lp))
    u_rho[:,1:-1] = u_temp
    u_rho[:,0] = u_temp[:,0]
    u_rho[:,-1] = u_temp[:,-1]

    # v interpolation
    [M,Lp] = np.squeeze(out_nc.variables['v_surf']).shape
    Mp = M+1
    Mm = M-1
    v_temp = 0.5*(np.squeeze(out_nc.variables['v_surf'])[1:M,:]+np.squeeze(out_nc.variables['v_surf'])[:Mm,:])
    v_rho = np.zeros((Mp,Lp))
    v_rho[1:-1,:] = v_temp
    v_rho[0,:] = v_temp[0,:]
    v_rho[-1,:] = v_temp[-1,:]


    return u_rho,v_rho

def numdate(dt,start):
    '''
    dt: array of time in numbers
    start: string of start time of dt, e.g., 'seconds since 2000-01-01', 
           'days since 2000-01-01'
    '''
    dtout = num2date(dt,start,only_use_python_datetimes=True,only_use_cftime_datetimes=False)
    return dtout

