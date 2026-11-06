#!/usr/bin/env python3
import os, json, math, argparse
import numpy as np

from ase.build import nanotube
from ase.optimize import FIRE
from ase.io import write
from ase.md.verlet import VelocityVerlet
from ase.calculators.lammpslib import LAMMPSlib

# Calculator (AIREBO)
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
        files=[potfile],          # <- important
        log_file="lammps.log",
        keep_alive=True,
    )

# Geometry scaling
def scale_axial(atoms, deps):
    cell = atoms.cell.copy()
    cell[2, 2] *= (1.0 + deps)
    atoms.set_cell(cell, scale_atoms=True)

# NVE MD with chunked sampling
def run_nve(atoms, dt_fs, t_relax_ps, sample_every=10):
    from ase import units
    dt = dt_fs * units.fs
    steps = int(round((t_relax_ps * 1000.0) / dt_fs))
    steps = max(1, steps)

    t_fs, Epot, Ekin, Etot, Temp = [], [], [], [], []
    done = 0
    dyn = VelocityVerlet(atoms, timestep=dt)

    while done < steps:
        n = min(sample_every, steps - done)
        dyn.run(n)
        done += n
        # sample once per chunk
        ep = atoms.get_potential_energy()
        ek = atoms.get_kinetic_energy()
        et = ep + ek
        T = atoms.get_temperature()
        t_fs.append(done * dt_fs)
        Epot.append(ep); Ekin.append(ek); Etot.append(et); Temp.append(T)

    return {
        "t_fs": np.array(t_fs),
        "Epot": np.array(Epot),
        "Ekin": np.array(Ekin),
        "Etot": np.array(Etot),
        "Temp": np.array(Temp),
    }

# Analysis helpers
def rolling_rms(x, win):
    if len(x) < win:
        return np.array([])
    c = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x * x, 0, 0.0))
    out = []
    for i in range(win, len(x) + 1):
        s = c[i] - c[i - win]
        s2 = c2[i] - c2[i - win]
        m = s / win
        var = max(s2 / win - m * m, 0.0)
        out.append(math.sqrt(var))
    return np.array(out)

def estimate_relaxation_time(run, dt_fs, window_fs=50.0, rms_thresh=1e-3, natoms=1):
    t = run["t_fs"]
    E = run["Etot"] / natoms
    if len(t) < 2:
        return t[-1] / 1000.0
    # detrend Etot per-atom
    z = np.polyfit(t, E, 1)
    resid = E - np.polyval(z, t)
    win = max(1, int(window_fs / dt_fs))
    rr = rolling_rms(resid, win)
    if rr.size == 0:
        return t[-1] / 1000.0
    for i in range(len(rr)):
        if np.all(rr[i:] < rms_thresh):
            return t[i + win - 1] / 1000.0  # fs -> ps
    return t[-1] / 1000.0

def energy_drift_ppm_per_ps(run, natoms=1):
    t_ps = run["t_fs"] / 1000.0
    y = run["Etot"] / natoms
    if len(t_ps) < 2:
        return 0.0
    slope, intercept = np.polyfit(t_ps, y, 1)
    meanE = np.mean(y)
    return (abs(slope) / max(abs(meanE), 1e-12)) * 1e6

def save_csv(path, run):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_fs", "Epot", "Ekin", "Etot", "Temp_K"])
        for a, b, c, d, e in zip(run["t_fs"], run["Epot"], run["Ekin"], run["Etot"], run["Temp"]):
            w.writerow([f"{a:.6f}", f"{b:.8f}", f"{c:.8f}", f"{d:.8f}", f"{e:.3f}"])

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.1, help="time step in fs (e.g. 0.1, 0.2, 0.5, 1.0)")
    ap.add_argument("--deps", type=float, default=0.005, help="strain increment per step (e.g. 0.005=0.5%)")
    ap.add_argument("--trelax", type=float, default=2.0, help="NVE relaxation time per step (ps)")
    ap.add_argument("--nsteps", type=int, default=12, help="number of strain increments")
    ap.add_argument("--ucells", type=int, default=8, help="CNT length (# translational unit cells)")
    ap.add_argument("--rms", type=float, default=1e-3, help="RMS threshold (eV/atom) to call relaxed")
    ap.add_argument("--wfs", type=float, default=50.0, help="rolling RMS window (fs)")
    ap.add_argument("--drift_ppm", type=float, default=50.0, help="energy drift limit (ppm/ps) for unstable")
    args = ap.parse_args()

    tag = f"dt{args.dt}_deps{args.deps}_trelax{args.trelax}_n{args.nsteps}"
    outdir = os.path.join("out42", tag)
    os.makedirs(outdir, exist_ok=True)

    # (5,5) CNT, periodic along z
    atoms = nanotube(5, 5, length=args.ucells, bond=1.42, vacuum=10.0, symbol="C")
    atoms.set_pbc((False, False, True))
    atoms.calc = build_calc()

    # 0 K relax
    FIRE(atoms, logfile=os.path.join(outdir, "min.log")).run(fmax=0.01)
    write(os.path.join(outdir, "cnt_min.xyz"), atoms)
    natoms = len(atoms)

    records = []
    total_strain = 0.0

    for k in range(1, args.nsteps + 1):
        # apply axial strain
        scale_axial(atoms, args.deps)
        total_strain += args.deps

        # NVE relax + analysis
        run = run_nve(atoms, args.dt, args.trelax, sample_every=10)
        rel_ps = estimate_relaxation_time(run, args.dt, args.wfs, args.rms, natoms)
        drift = energy_drift_ppm_per_ps(run, natoms)
        Tmax = float(np.max(run["Temp"]))
        meanT = float(np.mean(run["Temp"]))
        rate = args.deps / max(rel_ps, 1e-12)  # 1/ps

        unstable = (drift > args.drift_ppm) or (Tmax > 2000.0) or (rel_ps >= 0.95*args.trelax)

        # write per-step outputs
        save_csv(os.path.join(outdir, f"energies_step{k:02d}.csv"), run)
        write(os.path.join(outdir, f"geom_step{k:02d}.xyz"), atoms)

        records.append({
            "step": k,
            "deps": args.deps,
            "dt_fs": args.dt,
            "trelax_ps": args.trelax,
            "rel_time_ps": rel_ps,
            "rate_per_ps": rate,
            "E_drift_ppm_per_ps": drift,
            "Tmax_K": Tmax,
            "Tmean_K": meanT,
            "unstable": bool(unstable),
        })

        print(f"[step {k:02d}] relax≈{rel_ps:.3f} ps | rate≈{rate:.3e} 1/ps | "
              f"drift≈{drift:.1f} ppm/ps | Tmax≈{Tmax:.0f} K | unstable={unstable}")

        if unstable:
            print("⚠️  Instability detected — stopping further increments.")
            break

    # summary
    summary = {
        "params": vars(args),
        "natoms": natoms,
        "total_applied_strain": total_strain,
        "records": records,
        "notes": {
            "instability_flags": "energy drift > drift_ppm OR Tmax > 2000 K",
            "relaxation_time": "first time where rolling RMS(Etot per atom) < threshold thereafter",
        },
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Done. Outputs in: {outdir}")

if __name__ == "__main__":
    main()
