from htmd.ui import *

def LJ_potential(prot,lig):
    from scipy.spatial.distance import cdist
    pcoords = prot.coords.squeeze()
    ligcoords = lig.coords.squeeze()
    pl_dists = cdist(pcoords,ligcoords)
    return np.min(pl_dists)

crystal = Molecule('3iej.pdb')
prot = crystal.copy()
prot.filter('protein and chain A and noh')
prot.view(style='Licorice',color=8)

lig = crystal.copy()
lig.filter('chain A and resname 599')
lig.get_rot_bonds()
print('LJ:',LJ_potential(prot,lig))
lig.view()
lig._moveVMD('scalein')
lig._moveVMD('scaleout')
lig._moveVMD('nextdih')
lig._moveVMD('nextdih')
lig._moveVMD('switch_dir')


