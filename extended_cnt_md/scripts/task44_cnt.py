#!/usr/bin/env python3
import os, json, math, argparse, glob
import numpy as np
from ase import units
from ase.io import write
from ase.build import nanotube
from ase.optimize import FIRE
from ase.md.langevin import Langevin
from ase.calculators.lammpslib import LAMMPSlib
import matplotlib.pyplot as plt

# LAMMPS/AIREBO
def build_calc():
    pot = os.path.join(os.environ["LAMMPS_POTENTIALS"], "CH.airebo")
    lmpcmds = [
        "pair_style airebo 3.0",
        "pair_coeff * * CH.airebo C",
        "neighbor 2.0 bin",
        "neigh_modify delay 0 every 1 check yes",
    ]
    return LAMMPSlib(lmpcmds=lmpcmds, atom_types={"C": 1},
                     files=[pot], log_file="lammps_nvt.log", keep_alive=True)

# Geometry
def scale_axial(atoms, deps):
    cell = atoms.cell.copy()
    cell[2, 2] *= (1.0 + deps)
    atoms.set_cell(cell, scale_atoms=True)

# NVT run at 300 K
def run_nvt(atoms, dt_fs, t_relax_ps, T=300.0, tau_fs=100.0, sample_every=10):
    dt = dt_fs * units.fs
    steps = max(1, int(round((t_relax_ps * 1000.0) / dt_fs)))
    dyn = Langevin(atoms,
                   timestep=dt,
                   temperature_K=T,
                   friction=1.0 / (tau_fs * units.fs),  # gamma = 1/tau
                   rng=np.random.default_rng(12345))
    t_fs, Epot, Ekin, Etot, Temp = [], [], [], [], []
    done = 0
    while done < steps:
        n = min(sample_every, steps - done)
        dyn.run(n)
        done += n
        ep = atoms.get_potential_energy()
        ek = atoms.get_kinetic_energy()
        et = ep + ek
        Tinst = atoms.get_temperature()
        t_fs.append(done * dt_fs)
        Epot.append(ep); Ekin.append(ek); Etot.append(et); Temp.append(Tinst)
    return {k: np.array(v) for k, v in
            dict(t_fs=t_fs,Epot=Epot,Ekin=Ekin,Etot=Etot,Temp=Temp).items()}

# Helpers
def save_csv(path, run):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_fs","Epot","Ekin","Etot","Temp_K"])
        for a,b,c,d,e in zip(run['t_fs'],run['Epot'],run['Ekin'],run['Etot'],run['Temp']):
            w.writerow([f"{a:.6f}",f"{b:.8f}",f"{c:.8f}",f"{d:.8f}",f"{e:.3f}"])

def per_step_energies(outdir, natoms, deps, nsteps, per_atom=True):
    eps, Epl, Ekl, Etl = [], [], [], []
    for k in range(1, nsteps+1):
        csvp = os.path.join(outdir, f"energies_step{k:02d}.csv")
        if not os.path.exists(csvp):
            break
        a = np.genfromtxt(csvp, delimiter=",", names=True, dtype=None, encoding=None)
        if a.ndim == 0: a = np.array([a])
        el, ekl, etl = float(a["Epot"][-1]), float(a["Ekin"][-1]), float(a["Etot"][-1])
        if per_atom and natoms>0:
            el/=natoms; ekl/=natoms; etl/=natoms
        eps.append(k*deps); Epl.append(el); Ekl.append(ekl); Etl.append(etl)
    return np.array(eps), np.array(Epl), np.array(Ekl), np.array(Etl)

def main():
    ap = argparse.ArgumentParser(description="Task 4.4: NVT(300K) uniaxial stretch")
    ap.add_argument("--dt", type=float, default=0.1, help="time step (fs)")
    ap.add_argument("--deps", type=float, default=0.01, help="strain increment per step")
    ap.add_argument("--trelax", type=float, default=0.5, help="relax time per step (ps)")
    ap.add_argument("--nsteps", type=int, default=12, help="number of strain steps")
    ap.add_argument("--ucells", type=int, default=4, help="CNT length (unit cells)")
    ap.add_argument("--tau", type=float, default=100.0, help="thermostat time (fs)")
    ap.add_argument("--T", type=float, default=300.0, help="target temperature (K)")
    ap.add_argument("--compare_dir", default="", help="optional NVE out42 directory to overlay")
    args = ap.parse_args()

    tag = f"dt{args.dt}_deps{args.deps}_trelax{args.trelax}_n{args.nsteps}_T{int(args.T)}"
    outdir = os.path.join("out44", tag)
    os.makedirs(outdir, exist_ok=True)

    atoms = nanotube(5,5, length=args.ucells, bond=1.42, vacuum=10.0, symbol="C")
    atoms.set_pbc((False, False, True))
    atoms.calc = build_calc()

    # 0 K geometry relax once
    FIRE(atoms, logfile=os.path.join(outdir, "min.log")).run(fmax=0.01)
    write(os.path.join(outdir, "cnt_min.xyz"), atoms)
    natoms = len(atoms)

    records = []
    for k in range(1, args.nsteps+1):
        # strain step
        scale_axial(atoms, args.deps)

        # NVT @ 300 K relax
        run = run_nvt(atoms, dt_fs=args.dt, t_relax_ps=args.trelax,
                      T=args.T, tau_fs=args.tau, sample_every=10)

        # save step data
        save_csv(os.path.join(outdir, f"energies_step{k:02d}.csv"), run)
        write(os.path.join(outdir, f"geom_step{k:02d}.xyz"), atoms)

        Tmax = float(np.max(run["Temp"]))
        Tmean = float(np.mean(run["Temp"]))
        records.append({"step":k,"deps":args.deps,"dt_fs":args.dt,
                        "trelax_ps":args.trelax,"Tmax_K":Tmax,"Tmean_K":Tmean})
        print(f"[step {k:02d}] <T>≈{Tmean:.1f} K, T_max≈{Tmax:.1f} K")

    summary = {"params": vars(args), "natoms": natoms, "records": records}
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Compare to an NVE folder (out42/...)
    if args.compare_dir:
        nve_dir = args.compare_dir
    else:
        # pick NVE folder with most steps if available
        cand = glob.glob(os.path.join("out42","dt*_deps*_trelax*_n*"))
        nve_dir = ""
        if cand:
            cand = sorted(cand, key=lambda c: len(glob.glob(os.path.join(c,"energies_step*.csv"))))
            nve_dir = cand[-1]

    epsNVT, EplNVT, EklNVT, EtlNVT = per_step_energies(outdir, natoms, deps=args.deps, nsteps=args.nsteps)
    if nve_dir:
        with open(os.path.join(nve_dir,"summary.json")) as f:
            sNVE = json.load(f)
        epsNVE, EplNVE, EklNVE, EtlNVE = per_step_energies(nve_dir, sNVE["natoms"],
                                                           deps=float(sNVE["params"]["deps"]),
                                                           nsteps=int(sNVE["params"]["nsteps"]))
    else:
        EplNVE=EklNVE=EtlNVE=epsNVE=np.array([])

    # Plot: energies at end of relax vs strain (per atom)
    plt.figure()
    if epsNVE.size:
        plt.plot(epsNVE, EplNVE, 'o-', label="E_pot NVE (end)")
        plt.plot(epsNVE, EtlNVE, 'o-', label="E_tot NVE (end)")
    plt.plot(epsNVT, EplNVT, 's-', label="E_pot NVT@300K (end)")
    plt.plot(epsNVT, EklNVT, 's-', label="E_kin NVT@300K (end)")
    plt.plot(epsNVT, EtlNVT, 's-', label="E_tot NVT@300K (end)")
    plt.xlabel("Applied axial strain ε")
    plt.ylabel("Energy (eV/atom)")
    plt.title("Task 4.4: NVE vs NVT(300K) — energies vs strain")
    plt.legend(); plt.tight_layout()
    figp = os.path.join(outdir, "compare_NVE_vs_NVT.png")
    plt.savefig(figp, dpi=180)
    print(f"\n✅ Done. Outputs in: {outdir}\nSaved compare plot: {figp}")

if __name__ == "__main__":
    main()
