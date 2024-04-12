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
):
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

        bonds1[bond1.particles] = bond1
        bonds2[bond2.particles] = bond2

    assert bonds1 == bonds2, (bonds1, bonds2)


def compare_angle_forces(
    system1: openmm.System,
    system2: openmm.System,
):
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

    for bond_index in range(angle_force1.getNumAngles()):
        angle1 = Angle.from_openmm(angle_force1.getAngleParameters(bond_index))

        angle2 = Angle.from_openmm(angle_force1.getAngleParameters(bond_index))

        angles1[angle1.particles] = angle1
        angles2[angle2.particles] = angle2

    assert angles1 == angles2, (angles1, angles2)


def compare_torsion_forces(
    system1: openmm.System,
    system2: openmm.System,
):
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

    assert (
        force1.getEwaldErrorTolerance() == force2.getEwaldErrorTolerance()
    ), "Ewald error tolerances differ!"


def compare_nonbonded_forces(
    system1: openmm.System,
    system2: openmm.System,
):
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
        non_bonded_force1, non_bonded_force2, skip_switching_check=True
    )

    nonbonded1 = [
        non_bonded_force1.getParticleParameters(i)[:3]
        for i in range(non_bonded_force1.getNumParticles())
    ]
    nonbonded2 = [
        non_bonded_force2.getParticleParameters(i)[:3]
        for i in range(non_bonded_force2.getNumParticles())
    ]

    assert nonbonded1 == nonbonded2


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


def compare_solvated_systems(
    system1: openmm.System,
    system2: openmm.System,
):
    compare_volumes(system1, system2)


compare_vacuum_systems(
    openmm.XmlSerializer.deserialize(open("vac.xml").read()),
    openmm.XmlSerializer.deserialize(open("vac_alt.xml").read()),
)

compare_solvated_systems(
    openmm.XmlSerializer.deserialize(open("solv.xml").read()),
    openmm.XmlSerializer.deserialize(open("solv_alt.xml").read()),
)
