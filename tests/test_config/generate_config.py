#!/usr/bin/env python
# -*- Encoding: UTF-8 -*-

"""
Generate directories with input.ini files for
blob amplitude scan
"""

from twodads.twodads import input2d
import os

amplitudes = [1.0, 2.0, 5.0, 10.0]
inp = input2d()

inp['runnr'] = 1
inp['xleft'] = -25.0
inp['xright'] = +25.0
inp['ylow']	= -25.0
inp['yup'] = +25.0
inp['Nx'] = 1024
inp['My'] = 1024
inp['scheme'] = 'ss4'
inp['tlevs'] = 4
inp['deltat'] = 1e-3
inp['tend']	= 1e+1
inp['tdiag'] = 1e-2
inp['tout']	= 1e-1
inp['log_theta'] = 1
inp['do_particle_tracking']	= 0
inp['nprobes']	= 4
inp['theta_rhs'] = 'theta_rhs_log'
inp['omega_rhs'] = 'omega_rhs_ic'
inp['strmf_solver'] = 'inv_laplace'
inp['init_function'] = 'theta_gaussian'
inp['initial_conditions'] = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
inp['model_params']	= [0.001, 0.000, 1.0, 0.0, 0.0, 0.0]
inp['output'] =  ['theta', 'omega', 'strmf']
inp['diagnostics'] = ['blobs']
inp['nthreads']	= 1


for run, amp in enumerate(amplitudes):
    simdir = 'run%d' % (run)
    inp['runnr'] = run
    inp['initial_conditions'][1] = amp

    # See if directory exists. If not, create
    try:
        os.stat(simdir)
        print '%s exists already...' % simdir
    except:
        os.mkdir(simdir)
        print 'creating directory %s' % simdir

    inp.to_file(os.path.join(simdir, 'input.ini'))

# print '01 runnr = %d ' % inp['runnr']
# print '02 xleft = %f' % inp['xleft']
# print '03 xright = %f' % inp['xright']
# print '04 ylow = %f' % inp['ylow']
# print '05 yup = %f' % inp['yup']
# print '06 Nx = %d' % inp['Nx']
# print '07 My = %d' % inp['My']
# print '08 scheme = %s' % inp['scheme']
# print '09 tlevs = %d' % inp['tlevs']
# print '10 deltat = %f' % inp['deltat']
# print '11 tend = %f' % inp['tend']
# print '12 tdiag = %f' % inp['tdiag']
# print '13 tout = %f' % inp['tout']
# print '14 log_theta = %d' % inp['log_theta']
# print '15 do_particle_tracking = %d' % inp['do_particle_tracking']
# print '16 nprobes = %d' % inp['nprobes']
# print '17 theta_rhs = %s' % inp['theta_rhs']
# print '18 omega_rhs = %s' % inp['omega_rhs']
# print '19 strmf_solver = %s' % inp['strmf_solver']
# print '20 init_function = %s' % inp['init_function']
# print '21 initial_conditions = ' + ' '.join([str(i) for i in inp['initial_conditions']])
# print '22 model_params = ' + ' '.join([str(mp) for mp in inp['model_params']])
# print '23 output = ' + ' '.join(inp['output'])
# print '24 diagnostics = ' + ' '.join(inp['diagnostics'])
# print '25 nthreads = %d' % inp['nthreads']
#
# print '   deltax = %f' % inp['deltax']
# print '   deltay = %f' % inp['deltay']
# print '   Lx = %f' % inp['Lx']
# print '   Ly = %f' % inp['Ly']


# End ile generate_config.py
