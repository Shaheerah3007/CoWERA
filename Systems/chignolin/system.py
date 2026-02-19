# system.py
import simtk.unit as unit
import simtk.openmm as omm
import simtk.openmm.app as omma


def make_system(top, temp=275.0):
    """
    Create an implicit solvent OpenMM system (GB-HCT).

    Parameters
    ----------
    prmtop : openmm.app.AmberPrmtopFile
        Loaded AmberPrmtopFile object.

    temp : float, optional
        Temperature in Kelvin. Default is 275 K.

    Returns
    -------
    system : openmm.System
    integrator : openmm.LangevinMiddleIntegrator
    """

    if temp <= 0:
        raise ValueError("Temperature must be positive (Kelvin).")

    # -------------------------
    # Create implicit solvent system
    # -------------------------
    system = top.createSystem(
        nonbondedMethod=omma.NoCutoff,      # No periodic boundary conditions
        constraints=omma.HBonds,
        implicitSolvent=omma.HCT,           # igb=1 equivalent
        soluteDielectric=1.0,
        solventDielectric=78.5
    )

    # -------------------------
    # Langevin integrator
    # -------------------------
    integrator = omm.LangevinMiddleIntegrator(
        temp * unit.kelvin,
        5.0 / unit.picosecond,      # gamma_ln
        0.002 * unit.picoseconds    # 2 fs timestep
    )

    return system, integrator