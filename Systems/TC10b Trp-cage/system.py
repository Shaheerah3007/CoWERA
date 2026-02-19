# system.py

import simtk.unit as unit
import simtk.openmm as omm
import simtk.openmm.app as omma


def make_system(top, temp):
    """
    Create an OpenMM System and Integrator (NPT ensemble).

    Parameters
    ----------
    top : openmm.app.TopologyFile object
        Topology file object (e.g., GromacsTopFile or AmberPrmtopFile)
        Must support .createSystem()

    temp : float
        Temperature in Kelvin (scalar)

    Returns
    -------
    system : openmm.System
    integrator : openmm.LangevinMiddleIntegrator
    """

    if temp <= 0:
        raise ValueError("Temperature must be positive (Kelvin).")

    # -------------------------
    # Create system
    # -------------------------
    system = top.createSystem(
        nonbondedMethod=omma.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=omma.HBonds
    )

    # -------------------------
    # Add barostat (NPT)
    # -------------------------
    barostat = omm.MonteCarloBarostat(
        1.0 * unit.bar,
        temp * unit.kelvin
    )
    system.addForce(barostat)

    # -------------------------
    # Create integrator
    # -------------------------
    integrator = omm.LangevinMiddleIntegrator(
        temp * unit.kelvin,
        1.0 / unit.picosecond,       # friction
        0.002 * unit.picoseconds     # 2 fs timestep
    )

    return system, integrator