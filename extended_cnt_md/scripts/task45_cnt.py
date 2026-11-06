#!/usr/bin/env python3
import os, json, math, argparse
import numpy as np
import matplotlib.pyplot as plt

from ase.build import nanotube
from ase.optimize import FIRE
from ase import units
from ase.io import write
from ase.md.langevin import Langevin
from ase.calculators.lammpslib import LAMMPSlib

# Helpers

def build_calc():
    potfile = os.path.join(os.environ["LAMMPS_POTENTIALS"], "CH.airebo")
    lmpcmds = [
        "pair_style airebo 3.0",
        "pair_coeff * * CH.airebo C",
        "neighbor 2.0 bin",
        "neigh_modify delay 0 every 1 check yes",
    ]
    return LAMMPSlib(
        lmpcmds=lmpcmds,
        atom_types={"C": 1},
        files=[potfile],
        log_file="lammps.log",
        keep_alive=True,
    )

def axial_scale(atoms, deps):
    cell = atoms.cell.copy()
    cell[2, 2] *= (1.0 + deps)
    atoms.set_cell(cell, scale_atoms=True)

def mean_radius_xy(atoms):
    xy = atoms.get_positions()[:, :2]
    r = np.sqrt((xy**2).sum(axis=1))
    return float(np.mean(r))

def box_to_wall_stress_GPa(atoms, szz_box_GPa, t_wall=3.4):
    # box stress uses A_box = Lx * Ly; convert to wall stress F/A_eff, A_eff = 2*pi*R*t
    A_box = atoms.cell.lengths()[0] * atoms.cell.lengths()[1]  # Å^2
    R = mean_radius_xy(atoms)                                  # Å
    A_eff = 2.0 * math.pi * R * t_wall                         # Å^2
    if A_eff <= 1e-12:
        return szz_box_GPa
    return szz_box_GPa * (A_box / A_eff)

def run_nvt(atoms, T, dt_fs, trelax_ps, tauT_fs):
    if trelax_ps <= 0:
        return
    dyn = Langevin(
        atoms,
        timestep=dt_fs * units.fs,
        temperature_K=T,
        friction=1.0 / (tauT_fs * units.fs),
    )
    nsteps = max(1, int(round((trelax_ps * 1000.0) / dt_fs)))
    dyn.run(nsteps)

# Main deformation driver

def deform_cnt(
    dt_fs=0.5, deps=0.01, trelax_ps=0.3, nsteps=40, ucells=6,
    T=300.0, tauT_fs=100.0, wall_thick_A=3.4, fmax=0.01,
    stop_Tmax=4000.0, stop_drop_frac=0.2, outdir="task45_out"
):
    os.makedirs(outdir, exist_ok=True)

    # Build tube (periodic in all directions, large vacuum xy)
    atoms = nanotube(5, 5, length=ucells, bond=1.42, vacuum=12.0, symbol="C")
    atoms.set_pbc((True, True, True))
    atoms.calc = build_calc()

    # 0 K relax to get a clean reference
    FIRE(atoms, logfile=os.path.join(outdir, "min.log")).run(fmax=fmax)
    write(os.path.join(outdir, "cnt_min.xyz"), atoms)

    L0 = atoms.cell[2, 2]
    nat = len(atoms)

    # Small thermalization to ~300 K before the first deform
    if T > 0:
        run_nvt(atoms, T, dt_fs, trelax_ps, tauT_fs)

    records = []
    prev_wall_sigma = None
    fractured_at = None

    for k in range(1, nsteps + 1):
        # Apply strain jump
        axial_scale(atoms, deps)

        # Minimize lightly (geometry cleanup) then NVT relax at T
        FIRE(atoms).run(fmax=fmax)
        run_nvt(atoms, T, dt_fs, trelax_ps, tauT_fs)

        # Energies and temperature
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        etot = epot + ekin
        Tinst = atoms.get_temperature()

        # Stress in box units (ASE returns eV/Å^3; convert to GPa)
        s = atoms.get_stress(voigt=False)  # 3x3
        szz_box_GPa = -float(s[2, 2]) / units.GPa  # tension positive

        # Convert to wall stress
        szz_wall_GPa = box_to_wall_stress_GPa(atoms, szz_box_GPa, t_wall=wall_thick_A)

        # Engineering strain
        strain = (atoms.cell[2, 2] - L0) / L0

        records.append({
            "step": k,
            "strain": strain,
            "E_pot_eV": epot / nat,
            "E_kin_eV": ekin / nat,
            "E_tot_eV": etot / nat,
            "T_K": Tinst,
            "szz_box_GPa": szz_box_GPa,
            "szz_wall_GPa": szz_wall_GPa,
        })

        print(f"[{k:02d}] ε={strain:6.3f} | σ_zz(wall)={szz_wall_GPa:7.3f} GPa | "
              f"T≈{Tinst:6.1f} K | E_tot/atom={etot/nat: .4f} eV")

        # fracture heuristics: sharp stress drop or huge T
        if prev_wall_sigma is not None:
            if szz_wall_GPa < (1.0 - stop_drop_frac) * prev_wall_sigma:
                fractured_at = strain
                print("⚠️  Fracture detected by stress drop.")
                break
        if Tinst > stop_Tmax:
            fractured_at = strain
            print("⚠️  Fracture/instability detected by high temperature.")
            break

        prev_wall_sigma = szz_wall_GPa

        # Save a snapshot each step (optional)
        if k <= 3 or k % 5 == 0:
            write(os.path.join(outdir, f"geom_step{k:03d}.xyz"), atoms)

    # Persist data
    import csv
    csv_path = os.path.join(outdir, "stress_strain.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "strain", "E_pot_eV", "E_kin_eV", "E_tot_eV",
                    "T_K", "szz_box_GPa", "szz_wall_GPa"])
        for r in records:
            w.writerow([r["step"], r["strain"], r["E_pot_eV"], r["E_kin_eV"],
                        r["E_tot_eV"], r["T_K"], r["szz_box_GPa"], r["szz_wall_GPa"]])

    peak_idx = np.argmax([r["szz_wall_GPa"] for r in records]) if records else 0
    peak_sigma = records[peak_idx]["szz_wall_GPa"] if records else None
    peak_strain = records[peak_idx]["strain"] if records else None
    E_increase = (records[peak_idx]["E_tot_eV"] - records[0]["E_tot_eV"]) if records else None

    summary = {
        "params": dict(dt_fs=dt_fs, deps=deps, trelax_ps=trelax_ps, nsteps=nsteps,
                       ucells=ucells, T=T, tauT_fs=tauT_fs, wall_thick_A=wall_thick_A),
        "natoms": nat,
        "fractured_at_strain": fractured_at,
        "peak_wall_stress_GPa": peak_sigma,
        "peak_at_strain": peak_strain,
        "bond_breaking_energy_increase_eV_per_atom": E_increase,
        "records_count": len(records),
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    if records:
        eps = np.array([r["strain"] for r in records])
        sig = np.array([r["szz_wall_GPa"] for r in records])
        Epot = np.array([r["E_pot_eV"] for r in records])
        Etot = np.array([r["E_tot_eV"] for r in records])

        plt.figure(figsize=(8,6))
        plt.plot(eps, sig, "o-")
        plt.xlabel("Engineering strain ε")
        plt.ylabel("Axial wall stress σzz (GPa)")
        plt.title("CNT stress–strain (AIREBO, 300 K)")
        plt.grid(True, alpha=0.3)
        if fractured_at is not None:
            plt.axvline(fractured_at, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "stress_strain.png"), dpi=160)

        plt.figure(figsize=(8,6))
        plt.plot(eps, Etot, "o-", label="E_tot")
        plt.plot(eps, Epot, "o-", label="E_pot")
        plt.xlabel("Engineering strain ε")
        plt.ylabel("Energy (eV/atom)")
        plt.title("Energy vs strain (end of NVT relax)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "energy_vs_strain.png"), dpi=160)

    print(f"\n✅ Wrote: {csv_path} and {os.path.join(outdir,'summary.json')}")
    return records, summary

# CLI

def main():
    ap = argparse.ArgumentParser(description="Task 4.5 – CNT stress–strain with AIREBO at 300 K")
    ap.add_argument("--dt", type=float, default=0.5, help="MD timestep (fs)")
    ap.add_argument("--deps", type=float, default=0.01, help="strain increment per step")
    ap.add_argument("--trelax", type=float, default=0.3, help="NVT relax time per step (ps)")
    ap.add_argument("--nsteps", type=int, default=40, help="number of strain steps")
    ap.add_argument("--ucells", type=int, default=6, help="axial length in translational unit cells")
    ap.add_argument("--T", type=float, default=300.0, help="thermostat temperature (K)")
    ap.add_argument("--tauT", type=float, default=100.0, help="thermostat time constant (fs)")
    ap.add_argument("--out", type=str, default="task45_out", help="output directory")
    args = ap.parse_args()

    deform_cnt(
        dt_fs=args.dt, deps=args.deps, trelax_ps=args.trelax, nsteps=args.nsteps,
        ucells=args.ucells, T=args.T, tauT_fs=args.tauT, outdir=args.out
    )

if __name__ == "__main__":
    main()
