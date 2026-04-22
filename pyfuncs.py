import numpy as np
from netCDF4 import Dataset,num2date

def calc_ij(nc_grd,lat_sites,lon_sites):
    # find closest eta and xi points in grid
    # to given latitude/longitude

    lon_nc = nc_grd.variables['lon_rho'][:,:]
    # fixes if the longitude is 0 to 360 instead of -180 to 180
    lon_nc[lon_nc>180] -= 360

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

def rho_uv_angle(roms_file,grid_file,rotate=True):
    '''
    Compute u_rho and v_rho from ROMS u/v and optionally rotate
    to east/north.

    Parameters
    ----------
    roms_file : str
        ROMS NetCDF file
    rotate : bool
        If True, rotate to east/north using angle

    Returns
    -------
    u_rho, v_rho : ndarray
        If rotate=True → east/north velocities
        shape: (time, s_rho, eta_rho, xi_rho)
    '''

    out_nc = Dataset(roms_file, 'r')
    grd_nc = Dataset(grid_file, 'r')

    # --- Load variables (KEEP TIME DIMENSION) ---
    u = out_nc.variables['u'][:]   # (time, s, eta_u, xi_u)
    v = out_nc.variables['v'][:]   # (time, s, eta_v, xi_v)
    angle = grd_nc.variables['angle'][:]  # (eta_rho, xi_rho)

    # --- Dimensions ---
    Nt, Ns, Mp, L = u.shape
    Lp = L + 1

    # --- u → rho (interp in xi direction) ---
    u_temp = 0.5 * (u[:, :, :, 1:L] + u[:, :, :, :L-1])

    u_rho = np.zeros((Nt, Ns, Mp, Lp))
    u_rho[:, :, :, 1:-1] = u_temp
    u_rho[:, :, :, 0] = u_temp[:, :, :, 0]
    u_rho[:, :, :, -1] = u_temp[:, :, :, -1]

    # --- v → rho (interp in eta direction) ---
    Nt, Ns, M, Lp = v.shape
    Mp = M + 1

    v_temp = 0.5 * (v[:, :, 1:M, :] + v[:, :, :M-1, :])

    v_rho = np.zeros((Nt, Ns, Mp, Lp))
    v_rho[:, :, 1:-1, :] = v_temp
    v_rho[:, :, 0, :] = v_temp[:, :, 0, :]
    v_rho[:, :, -1, :] = v_temp[:, :, -1, :]

    # --- Rotation (optional) ---
    if rotate:
        # expand angle → (time, s, eta, xi)
        angle_4d = angle[np.newaxis, np.newaxis, :, :]

        cosang = np.cos(angle_4d)
        sinang = np.sin(angle_4d)

        u_east  = u_rho * cosang - v_rho * sinang
        v_north = u_rho * sinang + v_rho * cosang

        return u_east, v_north

    return u_rho, v_rho

def rho_uv_angle_surf(roms_file, grid_file, rotate=True):
    '''
    Compute surface u_rho and v_rho from ROMS u/v and optionally rotate
    to east/north. Returns 4D array compatible with vorticity function.

    Parameters
    ----------
    roms_file : str
        ROMS NetCDF file
    grid_file : str
        ROMS Grid NetCDF file
    rotate : bool
        If True, rotate to east/north using grid angle

    Returns
    -------
    u_rho, v_rho : ndarray
        If rotate=True -> east/north surface velocities
        shape: (time, s, eta_rho, xi_rho)  where s=1
    '''

    with Dataset(roms_file, 'r') as out_nc, Dataset(grid_file, 'r') as grd_nc:
        # Slicing with -1: extracts just the surface layer but KEEPS the 's' dimension
        # Resulting shape is 4D: (time, 1, eta_u, xi_u)
        u = out_nc.variables['u'][:, -1:, :, :]  
        v = out_nc.variables['v'][:, -1:, :, :]  
        angle = grd_nc.variables['angle'][:]  

    # --- u -> rho (interp in xi direction) ---
    Nt, Ns, Mp, L = u.shape
    Lp = L + 1
    
    u_temp = 0.5 * (u[:, :, :, 1:L] + u[:, :, :, :L-1])

    u_rho = np.zeros((Nt, Ns, Mp, Lp))
    u_rho[:, :, :, 1:-1] = u_temp
    u_rho[:, :, :, 0] = u_temp[:, :, :, 0]
    u_rho[:, :, :, -1] = u_temp[:, :, :, -1]

    # --- v -> rho (interp in eta direction) ---
    Nt, Ns, M, Lp = v.shape
    Mp = M + 1

    v_temp = 0.5 * (v[:, :, 1:M, :] + v[:, :, :M-1, :])

    v_rho = np.zeros((Nt, Ns, Mp, Lp))
    v_rho[:, :, 1:-1, :] = v_temp
    v_rho[:, :, 0, :] = v_temp[:, :, 0, :]
    v_rho[:, :, -1, :] = v_temp[:, :, -1, :]

    # --- Rotation (optional) ---
    if rotate:
        # Expand angle to 4D: (time, s, eta_rho, xi_rho)
        angle_4d = angle[np.newaxis, np.newaxis, :, :]

        cosang = np.cos(angle_4d)
        sinang = np.sin(angle_4d)

        u_east  = u_rho * cosang - v_rho * sinang
        v_north = u_rho * sinang + v_rho * cosang

        return u_east, v_north

    return u_rho, v_rho

def rho_uv_angle_bot(roms_file, grid_file, rotate=True):
    '''
    Compute bottom u_rho and v_rho from ROMS u/v and optionally rotate
    to east/north. Returns 4D array compatible with vorticity function.

    Parameters
    ----------
    roms_file : str
        ROMS NetCDF file
    grid_file : str
        ROMS Grid NetCDF file
    rotate : bool
        If True, rotate to east/north using grid angle

    Returns
    -------
    u_rho, v_rho : ndarray
        If rotate=True -> east/north surface velocities
        shape: (time, s, eta_rho, xi_rho)  where s=1
    '''

    with Dataset(roms_file, 'r') as out_nc, Dataset(grid_file, 'r') as grd_nc:
        # Slicing with -1: extracts just the surface layer but KEEPS the 's' dimension
        # Resulting shape is 4D: (time, 1, eta_u, xi_u)
        u = out_nc.variables['u'][:, 0:1, :, :]  
        v = out_nc.variables['v'][:, 0:1, :, :]  
        angle = grd_nc.variables['angle'][:]  

    # --- u -> rho (interp in xi direction) ---
    Nt, Ns, Mp, L = u.shape
    Lp = L + 1
    
    u_temp = 0.5 * (u[:, :, :, 1:L] + u[:, :, :, :L-1])

    u_rho = np.zeros((Nt, Ns, Mp, Lp))
    u_rho[:, :, :, 1:-1] = u_temp
    u_rho[:, :, :, 0] = u_temp[:, :, :, 0]
    u_rho[:, :, :, -1] = u_temp[:, :, :, -1]

    # --- v -> rho (interp in eta direction) ---
    Nt, Ns, M, Lp = v.shape
    Mp = M + 1

    v_temp = 0.5 * (v[:, :, 1:M, :] + v[:, :, :M-1, :])

    v_rho = np.zeros((Nt, Ns, Mp, Lp))
    v_rho[:, :, 1:-1, :] = v_temp
    v_rho[:, :, 0, :] = v_temp[:, :, 0, :]
    v_rho[:, :, -1, :] = v_temp[:, :, -1, :]

    # --- Rotation (optional) ---
    if rotate:
        # Expand angle to 4D: (time, s, eta_rho, xi_rho)
        angle_4d = angle[np.newaxis, np.newaxis, :, :]

        cosang = np.cos(angle_4d)
        sinang = np.sin(angle_4d)

        u_east  = u_rho * cosang - v_rho * sinang
        v_north = u_rho * sinang + v_rho * cosang

        return u_east, v_north

    return u_rho, v_rho

def rho_uv_angle_tind(roms_file,grid_file,tind,rotate=True):
    '''
    Compute u_rho and v_rho from ROMS u/v and optionally rotate
    to east/north.

    Parameters
    ----------
    roms_file : str
        ROMS NetCDF file
    rotate : bool
        If True, rotate to east/north using angle

    Returns
    -------
    u_rho, v_rho : ndarray
        If rotate=True → east/north velocities
        shape: (time, s_rho, eta_rho, xi_rho)
    '''

    out_nc = Dataset(roms_file, 'r')
    grd_nc = Dataset(grid_file, 'r')

    # --- Load variables (KEEP TIME DIMENSION) ---
    u = out_nc.variables['u'][tind,:,:,:]   # (time, s, eta_u, xi_u)
    v = out_nc.variables['v'][tind,:,:,:]   # (time, s, eta_v, xi_v)
    angle = grd_nc.variables['angle'][:]  # (eta_rho, xi_rho)

    # --- Dimensions ---
    Ns, Mp, L = u.shape
    Lp = L + 1

    # --- u → rho (interp in xi direction) ---
    u_temp = 0.5 * (u[:, :, 1:L] + u[:, :, :L-1])

    u_rho = np.empty((Ns, Mp, Lp))
    u_rho[:, :, 1:-1] = u_temp
    u_rho[:, :, 0] = u_temp[:, :, 0]
    u_rho[:, :, -1] = u_temp[:, :, -1]

    # --- v → rho (interp in eta direction) ---
    Ns, M, Lp = v.shape
    Mp = M + 1

    v_temp = 0.5 * (v[:, 1:M, :] + v[:, :M-1, :])

    v_rho = np.empty((Ns, Mp, Lp))
    v_rho[:, 1:-1, :] = v_temp
    v_rho[:, 0, :] = v_temp[:, 0, :]
    v_rho[:, -1, :] = v_temp[:, -1, :]

    # --- Rotation (optional) ---
    if rotate:
        # expand angle → (s, eta, xi)
        angle_3d = angle[np.newaxis, :, :]

        cosang = np.cos(angle_3d)
        sinang = np.sin(angle_3d)

        u_east  = u_rho * cosang - v_rho * sinang
        v_north = u_rho * sinang + v_rho * cosang

        return u_east, v_north

    return u_rho, v_rho

def vorticity(grd_file,u_rho,v_rho):
    '''
    Relative vorticity using forward/backward differences
    and nearest-value boundary fill.

    Parameters
    ----------
    roms_file : str
        ROMS NetCDF file (for pm, pn)
    u_rho, v_rho : ndarray
        (time, s, eta_rho, xi_rho)  already rotated

    Returns
    -------
    zeta : ndarray
        Relative vorticity (time, s, eta_rho, xi_rho)
    '''

    nc = Dataset(grd_file, 'r')

    pm = nc.variables['pm'][:]   # (eta, xi)
    pn = nc.variables['pn'][:]   # (eta, xi)

    Nt, Ns, Mp, Lp = u_rho.shape

    # Expand metrics
    pm4 = pm[np.newaxis, np.newaxis, :, :]
    pn4 = pn[np.newaxis, np.newaxis, :, :]

    # --- dv/dx ---
    dvdx = np.zeros_like(v_rho)

    # forward difference (interior)
    dvdx[:, :, :, :-1] = (v_rho[:, :, :, 1:] - v_rho[:, :, :, :-1]) * pm4[:, :, :, :-1]

    # backward at last column
    dvdx[:, :, :, -1] = (v_rho[:, :, :, -1] - v_rho[:, :, :, -2]) * pm4[:, :, :, -1]

    # --- du/dy ---
    dudy = np.zeros_like(u_rho)

    # forward difference (interior)
    dudy[:, :, :-1, :] = (u_rho[:, :, 1:, :] - u_rho[:, :, :-1, :]) * pn4[:, :, :-1, :]

    # backward at last row
    dudy[:, :, -1, :] = (u_rho[:, :, -1, :] - u_rho[:, :, -2, :]) * pn4[:, :, -1, :]

    # --- vorticity ---
    zeta = dvdx - dudy

    # --- optional: copy nearest values at boundaries (extra smoothing) ---
    # left/right edges
    zeta[:, :, :, 0]  = zeta[:, :, :, 1]
    zeta[:, :, :, -1] = zeta[:, :, :, -2]

    # top/bottom edges
    zeta[:, :, 0, :]  = zeta[:, :, 1, :]
    zeta[:, :, -1, :] = zeta[:, :, -2, :]

    return zeta

def numdate(dt,start):
    '''
    dt: array of time in numbers
    start: string of start time of dt, e.g., 'seconds since 2000-01-01', 
           'days since 2000-01-01'
    '''
    dtout = num2date(dt,start,only_use_python_datetimes=True,only_use_cftime_datetimes=False)
    return dtout

