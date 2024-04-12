import numpy
import openmm.unit
from openff.models.models import DefaultModel
from openff.units import Quantity
from openff.units.openmm import from_openmm
from pydantic.v1 import validator


def compare_thermostats(
    thermo1: openmm.AndersenThermostat,
    thermo2: openmm.AndersenThermostat,
):
    assert thermo1.getDefaultTemperature() == thermo2.getDefaultTemperature()
    assert (
        thermo1.getDefaultCollisionFrequency() == thermo2.getDefaultCollisionFrequency()
    )


def compare_barostats(
    thermo1: openmm.MonteCarloBarostat,
    thermo2: openmm.MonteCarloBarostat,
):
    assert thermo1.getDefaultPressure() == thermo2.getDefaultPressure()
    assert thermo1.getDefaultTemperature() == thermo2.getDefaultTemperature()


def compare_solvated_systems(
    system1: openmm.System,
    system2: openmm.System,
):
    # just assume there is only one barostat, only one thermostat
    system1_barostat = tuple(
        force for force in system1.getForces() if "Barostat" in force.__class__.__name__
    )[0]
    system2_barostat = tuple(
        force for force in system2.getForces() if "Barostat" in force.__class__.__name__
    )[0]

    compare_barostats(system1_barostat, system2_barostat)

    try:
        system1_thermostat = tuple(
            force
            for force in system1.getForces()
            if "Thermostat" in force.__class__.__name__
        )[0]
        system2_thermostat = tuple(
            force
            for force in system2.getForces()
            if "Thermostat" in force.__class__.__name__
        )[0]

        compare_thermostats(system1_thermostat, system2_thermostat)
    except IndexError:
        print("Didn't find thermostat(s)")


class ValenceParameter(DefaultModel):
    particles: tuple[int, ...]
    distance: openmm.unit.Quantity

    @validator("particles")
    def sort_particles(cls, v):
        return tuple(sorted(v))

    @classmethod
    def from_openmm(cls, data):
        return cls(
            particles=data[:-1],
            distance=data[-1],
        )


class Constraint(ValenceParameter):
    pass


class Bond(ValenceParameter):
    k: openmm.unit.Quantity

    @classmethod
    def from_openmm(cls, data):
        return cls(
            particles=data[:-2],
            distance=data[-2],
            k=data[-1],
        )


class Angle(Bond):
    pass


class Torsion(DefaultModel):
    particles: tuple[int, ...]
    periodicity: int
    phase: openmm.unit.Quantity
    k: openmm.unit.Quantity

    @validator("particles")
    def sort_particles(cls, v):
        return tuple(sorted(v))

    @classmethod
    def from_openmm(cls, data):
        return cls(
            particles=data[:-2],
            periodicity=data[4],
            phase=data[5],
            k=data[6],
        )


def compare_masses(
    system1: openmm.System,
    system2: openmm.System,
):
    assert (
        system1.getNumParticles() == system2.getNumParticles()
    ), "Number of particles differ"

    for particle_index in range(system1.getNumParticles()):
        assert system1.getParticleMass(particle_index) == system2.getParticleMass(
            particle_index
        )


def compare_constraints(
    system1: openmm.System,
    system2: openmm.System,
):
    assert (
        system1.getNumConstraints() == system2.getNumConstraints()
    ), "Number of constraints differ"

    for constraint_index in range(system1.getNumConstraints()):
        constraint1 = Constraint.from_openmm(
            system1.getConstraintParameters(constraint_index)
        )

        constraint2 = Constraint.from_openmm(
            system2.getConstraintParameters(constraint_index)
        )

        assert constraint1 == constraint2


def compare_bond_forces(
    system1: openmm.System,
    system2: openmm.System,
    particle_range: None | list[int] = None,
):
    if particle_range is None:
        particle_range = list(range(system1.getNumParticles()))

    bond_force1 = [
        force
        for force in system1.getForces()
        if isinstance(force, openmm.HarmonicBondForce)
    ][0]
    bond_force2 = [
        force
        for force in system2.getForces()
        if isinstance(force, openmm.HarmonicBondForce)
    ][0]

    assert bond_force1.getNumBonds() == bond_force2.getNumBonds()

    bonds1 = dict()
    bonds2 = dict()

    for bond_index in range(bond_force1.getNumBonds()):
        bond1 = Bond.from_openmm(bond_force1.getBondParameters(bond_index))

        bond2 = Bond.from_openmm(bond_force2.getBondParameters(bond_index))

        if any([index not in particle_range for index in bond1.particles]):
            continue

        if any([index not in particle_range for index in bond2.particles]):
            continue

        bonds1[bond1.particles] = bond1
        bonds2[bond2.particles] = bond2

    assert bonds1 == bonds2, (bonds1, bonds2)


def compare_angle_forces(
    system1: openmm.System,
    system2: openmm.System,
    particle_range: None | list[int] = None,
):
    if particle_range is None:
        particle_range = list(range(system1.getNumParticles()))

    angle_force1 = [
        force
        for force in system1.getForces()
        if isinstance(force, openmm.HarmonicAngleForce)
    ][0]
    angle_force2 = [
        force
        for force in system2.getForces()
        if isinstance(force, openmm.HarmonicAngleForce)
    ][0]

    assert angle_force1.getNumAngles() == angle_force2.getNumAngles()

    angles1 = dict()
    angles2 = dict()

    for angle_index in range(angle_force1.getNumAngles()):
        angle1 = Angle.from_openmm(angle_force1.getAngleParameters(angle_index))

        angle2 = Angle.from_openmm(angle_force1.getAngleParameters(angle_index))

        if any([index not in particle_range for index in angle1.particles]):
            continue

        if any([index not in particle_range for index in angle2.particles]):
            continue

        angles1[angle1.particles] = angle1
        angles2[angle2.particles] = angle2

    assert angles1 == angles2, (angles1, angles2)


def compare_torsion_forces(
    system1: openmm.System,
    system2: openmm.System,
    particle_range: None | list[int] = None,
):
    if particle_range is None:
        particle_range = list(range(system1.getNumParticles()))

    torsion_force1 = [
        force
        for force in system1.getForces()
        if isinstance(force, openmm.PeriodicTorsionForce)
    ][0]
    torsion_force2 = [
        force
        for force in system2.getForces()
        if isinstance(force, openmm.PeriodicTorsionForce)
    ][0]

    assert torsion_force1.getNumTorsions() == torsion_force2.getNumTorsions()

    torsions1 = dict()
    torsions2 = dict()

    for torsion_index in range(torsion_force1.getNumTorsions()):
        torsion1 = Torsion.from_openmm(
            torsion_force1.getTorsionParameters(torsion_index)
        )

        torsion2 = Torsion.from_openmm(
            torsion_force2.getTorsionParameters(torsion_index)
        )

        if any([index not in particle_range for index in torsion1.particles]):
            continue

        if any([index not in particle_range for index in torsion2.particles]):
            continue

        torsions1[torsion1.particles] = torsion1
        torsions2[torsion2.particles] = torsion2

    assert torsions1 == torsions2


def compare_nonbonded_settings(
    force1: openmm.NonbondedForce,
    force2: openmm.NonbondedForce,
    skip_switching_check: bool = False,
):
    assert force1.getCutoffDistance() == force2.getCutoffDistance(), "Cutoffs differ!"

    if not skip_switching_check:
        assert force1.getUseSwitchingFunction() == force2.getUseSwitchingFunction()
        assert (
            force1.getSwitchingDistance() == force2.getSwitchingDistance()
        ), "Switching distances differ!"

    if force1.getEwaldErrorTolerance() != force2.getEwaldErrorTolerance():
        print(
            "Ewald error tolerances differ:\n"
            f"\t{force1.getEwaldErrorTolerance()=}\n"
            f"\t{force2.getEwaldErrorTolerance()=}"
        )

        # TODO: Unclear if this should be an error


def compare_nonbonded_forces(
    system1: openmm.System,
    system2: openmm.System,
    particle_range: None | list[int] = None,
    allow_unequal_sigma: bool = False,
):
    if particle_range is None:
        particle_range = list(range(system1.getNumParticles()))

    non_bonded_force1 = [
        force
        for force in system1.getForces()
        if isinstance(force, openmm.NonbondedForce)
    ][0]

    non_bonded_force2 = [
        force
        for force in system2.getForces()
        if isinstance(force, openmm.NonbondedForce)
    ][0]

    compare_nonbonded_settings(
        non_bonded_force1,
        non_bonded_force2,
        skip_switching_check=True,
    )

    nonbonded1 = [
        non_bonded_force1.getParticleParameters(i)[:3]
        for i in range(non_bonded_force1.getNumParticles())
        if i in particle_range
    ]
    nonbonded2 = [
        non_bonded_force2.getParticleParameters(i)[:3]
        for i in range(non_bonded_force2.getNumParticles())
        if i in particle_range
    ]

    try:
        assert nonbonded1 == nonbonded2
    except AssertionError as error:
        if not allow_unequal_sigma:
            raise error
        else:
            nonbonded1 = [[charge, epsilon] for charge, _, epsilon in nonbonded1]

            nonbonded2 = [[charge, epsilon] for charge, _, epsilon in nonbonded2]

            assert nonbonded1 == nonbonded2


def compare_vacuum_systems(
    system1: openmm.System,
    system2: openmm.System,
):
    compare_masses(system1, system2)
    compare_constraints(system1, system2)
    compare_bond_forces(system1, system2)
    compare_angle_forces(system1, system2)
    compare_torsion_forces(system1, system2)
    compare_nonbonded_forces(system1, system2)


def _get_volume(system: openmm.System) -> Quantity:
    return numpy.prod(
        numpy.diagonal(
            from_openmm(system.getDefaultPeriodicBoxVectors()).m_as("nanometer")
        )
    )


def compare_volumes(
    system1: openmm.System,
    system2: openmm.System,
):
    volume1 = _get_volume(system1)
    volume2 = _get_volume(system2)

    try:
        assert (
            abs((volume1 - volume2) / volume1) < 0.1
        ), "Volumes differ by more than 10%"
    except AssertionError as error:
        # default setting doesn't actually pad, it just sets the box vectors
        # as twice the padding, so also compare densities
        density1 = system1.getNumParticles() / volume1
        density2 = system2.getNumParticles() / volume2

        assert (
            abs((density1 - density2) / density1) < 0.1
        ), "Densities differ by more than 10%"

        # TODO: re-raise this error since, density matching is not
        # (alone) good enough
        print(
            f"{error}, but densities are {density1:.2f} and "
            f"{density2:.2f} particles/nm^3"
        )


def find_number_ligand_atoms(system: openmm.System) -> int:
    """Really hacky way to guestimate when the ligand stops and waters start."""
    masses = numpy.asarray(
        [system.getParticleMass(i)._value for i in range(system.getNumParticles())]
    )

    for particle_index, mass in enumerate(masses):
        # check if the mass is close to that of an oxygen atom
        if abs(mass - 15.999) > 1e-3:
            continue

        # and do the same for the next 10 atoms
        if numpy.any(numpy.abs(masses[particle_index::3][:10] - 15.99943) > 1e-3):
            continue

        # same, offset by 1 and 2, which should be hydrogen
        if numpy.any(numpy.abs(masses[particle_index + 1 :: 3][:10] - 1.008) > 1e-3):
            continue

        if numpy.any(numpy.abs(masses[particle_index + 2 :: 3][:10] - 1.008) > 1e-3):
            continue

        return particle_index

    raise Exception("Couldn't find 10 water atoms, ordered OHH, after a ligand")


def compare_solvated_systems(
    system1: openmm.System,
    system2: openmm.System,
):
    compare_volumes(system1, system2)

    number_ligand_atoms = find_number_ligand_atoms(system1)

    if number_ligand_atoms != find_number_ligand_atoms(system2):
        raise Exception("Number of ligand atoms differ")

    # ligand forces
    ligand_particle_range = list(range(number_ligand_atoms))

    compare_bond_forces(system1, system2, ligand_particle_range)
    compare_angle_forces(system1, system2, ligand_particle_range)
    compare_torsion_forces(system1, system2, ligand_particle_range)

    compare_nonbonded_forces(system1, system2, ligand_particle_range)

    for particle_index in range(number_ligand_atoms, system1.getNumParticles()):
        if round(system1.getParticleMass(particle_index)._value) not in (1, 16):
            break

    water_particle_range = list(range(number_ligand_atoms, particle_index))

    # allow sigmas to be different, since they don't matter
    compare_nonbonded_forces(
        system1, system2, water_particle_range, allow_unequal_sigma=True
    )


compare_vacuum_systems(
    openmm.XmlSerializer.deserialize(open("vac.xml").read()),
    openmm.XmlSerializer.deserialize(open("vac_alt.xml").read()),
)

compare_solvated_systems(
    openmm.XmlSerializer.deserialize(open("solv.xml").read()),
    openmm.XmlSerializer.deserialize(open("solv_alt.xml").read()),
)
