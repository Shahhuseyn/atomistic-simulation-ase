#!/usr/bin/env python3
import os, glob, json, argparse
import numpy as np
import matplotlib.pyplot as plt

def best_outdir():
    cand = glob.glob(os.path.join("out42", "dt*_deps*_trelax*_n*"))
    if not cand:
        raise SystemExit("No out42/* folders found. Run task42_cnt.py first.")
    # choose the one with the most energies_step*.csv
    counts = [(len(glob.glob(os.path.join(c, "energies_step*.csv"))), c) for c in cand]
    counts.sort()
    return counts[-1][1]  # folder with max files

def load_summary(outdir):
    with open(os.path.join(outdir, "summary.json")) as f:
        return json.load(f)

def read_energies(csvp):
    a = np.genfromtxt(csvp, delimiter=",", names=True, dtype=None, encoding=None)
    if a.ndim == 0:
        a = np.array([a])
    return a

def per_step_values(outdir, per_atom=True):
    summ = load_summary(outdir)
    natoms = summ["natoms"]
    deps   = float(summ["params"]["deps"])
    nsteps = int(summ["params"]["nsteps"])

    strains, Epf, Ekf, Etf, Epl, Ekl, Etl = [], [], [], [], [], [], []
    found = 0
    for k in range(1, nsteps+1):
        csvp = os.path.join(outdir, f"energies_step{k:02d}.csv")
        if not os.path.exists(csvp):
            break
        a = read_energies(csvp)
        if a.size == 0:
            continue
        ef, ekf, etf = float(a["Epot"][0]),  float(a["Ekin"][0]),  float(a["Etot"][0])
        el, ekl, etl = float(a["Epot"][-1]), float(a["Ekin"][-1]), float(a["Etot"][-1])
        if per_atom and natoms > 0:
            ef/=natoms; ekf/=natoms; etf/=natoms
            el/=natoms; ekl/=natoms; etl/=natoms
        Epf.append(ef); Ekf.append(ekf); Etf.append(etf)
        Epl.append(el); Ekl.append(ekl); Etl.append(etl)
        strains.append(k*deps)
        found += 1

    return (np.array(strains),
            np.array(Epf), np.array(Ekf), np.array(Etf),
            np.array(Epl), np.array(Ekl), np.array(Etl),
            summ, found)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", help="explicit out42/<tag> directory to plot")
    args = ap.parse_args()

    outdir = args.outdir if args.outdir else best_outdir()
    print("Plotting from:", outdir)

    (eps, Epf, Ekf, Etf, Epl, Ekl, Etl, summ, nfound) = per_step_values(outdir, per_atom=True)
    if nfound == 0 or eps.size == 0:
        raise SystemExit("No step data found to plot in this folder.")

    # 1) energies at end of relax vs strain
    plt.figure()
    plt.plot(eps, Epl, 'o-', label="E_pot (relaxed end)")
    plt.plot(eps, Ekl, 'o-', label="E_kin (relaxed end)")
    plt.plot(eps, Etl, 'o-', label="E_tot (relaxed end)")
    plt.xlabel("Applied axial strain ε")
    plt.ylabel("Energy (eV/atom)")
    plt.title("CNT NVE stretch: energies vs strain (end of relax)")
    plt.legend(); plt.tight_layout()
    fig1 = os.path.join(outdir, "energies_vs_strain.png")
    plt.savefig(fig1, dpi=180)

    # 2) Etot jump vs relaxed
    plt.figure()
    plt.plot(eps, Etf, 'o--', label="E_tot (just after deform)")
    plt.plot(eps, Etl, 'o-',  label="E_tot (end of relax)")
    plt.xlabel("Applied axial strain ε")
    plt.ylabel("Total energy (eV/atom)")
    plt.title("Work input per increment (jump) and NVE conserve")
    plt.legend(); plt.tight_layout()
    fig2 = os.path.join(outdir, "Etot_jump_vs_relaxed.png")
    plt.savefig(fig2, dpi=180)

    print(f"Saved:\n  {fig1}\n  {fig2}\n  steps plotted: {nfound}")

if __name__ == "__main__":
    main()
