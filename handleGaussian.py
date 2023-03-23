import glob
import os.path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import argparse


# =============================================================================
def parse_arguments():

    desc = """TODO"""

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("-d", "--dir", dest="dir",
                        help="Directory to locate the logs from Gaussian.",
                        action="store", required=True, default=None)

    parser.add_argument("--dropenergy", dest="dropenergy", type=float,
                        help="Drop delta energy in kcal/mol.",
                        action="store", required=True, default=None)

    parser.add_argument("-l", "--labels", dest="labels", nargs="+",
                        help="List of labels in Gaussian log.",
                        action="store", required=True, default=None)

    parser.add_argument("-n", "--names", dest="names", nargs="+",
                        help="Names of the labels.",
                        action="store", required=True, default=None)

    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print("ERROR: Directory {0:s} does not exist.".format(args.dir))
        exit()

    if len(args.names) != len(args.labels):
        print("ERROR: {} and {} lists must have the same length".format(args.names, args.labels))
        exit()

    return args


# ===========================================================================================
def extract_energy_frozen(filename, label_to_find):

    # ====== FIND FIRST VALUE CONTAINING label_to_find
    line_found_first = ""
    line_found_last = ""
    with open(filename, 'r') as f:
        data = f.readlines()
        for idx in range(0, len(data)):
            if data[idx].count(label_to_find) != 0:
                line_found_first = data[idx]
                if data[idx].count("SCF Done:") != 0:
                    line_found_first = ""
                break
        for idx in range(len(data) - 1, -1, -1):
            if data[idx].count(label_to_find) != 0:
                line_found_last = data[idx]
                break
        lines = [line_found_first, line_found_last]
    extract_values = []
    for iline in lines:
        for item in iline.split():
            try:
                extract_values.append(float(item))
                break
            except ValueError:
                continue

    return extract_values


# ===========================================================================================
def extract_opt_com(ifile):

    with open(ifile, 'r') as finp:
        lines = finp.readlines()
        size = []
        for iline in lines:
            if iline.count("NAtoms") != 0:
                natoms = int(iline.split()[1])
        l_tmp = []
        for idx, iline in enumerate(lines):
            if iline.count("Standard orientation") != 0:
                l_tmp.append(idx)

    # print(ifile, l_tmp, l_tmp[-1])
    # linea = (ifile, l_tmp, l_tmp[-1])
    # print(linea)
    linea5 = (l_tmp[-1] + 5)
    linea45 = (l_tmp[-1] + 5 + natoms)
    # print(ifile)
    # print("Number of atoms: {}".format(magn))
    # print("StartMatrix: {}".format(linea5))
    # print("EndMatrix: {}".format(linea45))
    Matrix = list()
    for iline in lines[linea5:linea45]:

        # natom = float(iline.split()[0])
        typeatom = float(iline.split()[1])
        if typeatom == 1: typeatom = "H"
        if typeatom == 6: typeatom = "C"
        if typeatom == 8: typeatom = "O"
        xline = float(iline.split()[3])
        yline = float(iline.split()[4])
        zline = float(iline.split()[5])
        Matrix.append([typeatom, xline, yline, zline])

        # print(Matrix)
    # # with open(ifile+".com", 'w') as fout:
    ifile_tmp = os.path.split(ifile)[-1]
    ifile_newname = os.path.splitext(ifile_tmp)[0]+"_opt.com"
    ifile_newchk = os.path.splitext(ifile_tmp)[0] + "_opt.chk"
    line = "%chk={}\n".format(ifile_newchk)
    line += "%mem=2000MB\n"
    line += "%nproc=8\n"
    line += "#p opt M062X/6-311g(d,p)\n"
    line += "\n"
    line += "Optimization {}\n".format(ifile)
    line += "\n"
    line += "0 1\n"

    with open(ifile_newname, 'w') as fout:
        fout.writelines(line)
        for item in Matrix:
            linexyz = "{0:s} {1:f} {2:f} {3:f}\n".format(item[0], item[1], item[2], item[3])
            fout.writelines(linexyz)
        fout.writelines("\n")


# ===========================================================================================
def generate_sh_slurm(ifile):

    line1 = "#!/bin/bash\n"
    line1 += "#SBATCH --partition=generic\n"
    line1 += '#SBATCH --exclude=""\n'
    line1 += "#SBATCH --cpus-per-task=8\n"
    line1 += "#SBATCH --mem=2000M\n"
    line3 = "g16legacy_root=/dragofs/sw/campus/0.2/software/gaussian/16.gaussian/gaussian/g16_legacy\n"
    line3 += 'GAUSS_SCRDIR="$TMPDIR"\n'
    line3 += "source $g16legacy_root/bsd/g16.profile\n"
    line3 += "export g16legacy_root GAUSS_SCRDIR\n"

    ifile = os.path.split(ifile)[-1]
    ifile_sh = os.path.splitext(ifile)[0] + "_solvent.sh"
    ifile_com = os.path.splitext(ifile)[0] + "_solvent.com"

    with open(ifile_sh, 'w') as fsh:
        fsh.writelines(line1)
        numbers = ifile.split("_")
        line2 = "#SBATCH --job-name={0:s}_{1:s}\n".format(numbers[-2], numbers[-1])
        fsh.writelines(line2)
        fsh.writelines(line3)
        line4 = "$g16legacy_root/g16 {0:s}\n".format(ifile_com)
        fsh.writelines(line4)


# ===========================================================================================
def generate_bashscript_send_slurm(localdir=".", maxjobsslurm=50):

    """Generate the script **full_send.sh** in order to send jobs to a SLURM server

    Args:
        localdir (str): Path to store files in the local server.
        maxjobsslurm (str): Number of maximum jobs send to slurm

    """

    localfile = localdir+"/"+"full_send.sh"
    with open(localfile, 'w') as f:
        ll = '#!/bin/bash\n'
        ll += "\n"
        ll += "# NJOBS          --> Number of jobs sent to the slurm system\n"
        ll += "# MAXJOBSINSLURM --> Maximum number of jobs in the slurm system\n"
        ll += "# JOBSEND        --> Number of jobs finished in the slurm system\n"
        ll += "# TOTALJOBS      --> Jobs to be sent to the slurm system\n"
        ll += "# jobs.txt       --> Info of the jobs sent or finished in the slurm system\n"
        ll += '\n'
        ll += 'MAXJOBSINSLURM={}\n'.format(maxjobsslurm)
        ll += '\n'
        ll += 'NJOBS=`squeue -h |wc -ll`\n'
        ll += '\n'
        ll += 'COM=(`ls *.com`)\n'
        ll += 'LENGTH=${#COM[@]}\n'
        ll += '\n'

        ll += 'if [[ ! -e ./jobs.txt ]]; then\n'
        ll += '    echo -n >./jobs.txt\n'
        ll += 'fi\n'
        ll += '\n'
        ll += 'index=0\n'
        ll += 'while [ ${index} -lt ${LENGTH} ]; do\n'
        ll += '\n'
        ll += '    current=${COM[$index]}\n'
        ll += '\n'
        ll += '    if [[ $NJOBS -lt $MAXJOBSINSLURM ]]; then\n'
        ll += '        base="${COM[$index]%.*}"\n'
        ll += '        sbatch ${base}.sh  1 > tmp.txt\n'
        ll += '        jobid=`awk \'{print $NF}\' tmp.txt`\n'
        ll += '        echo "${jobid} ${base} ${base}.log" >>./jobs.txt\n'
        ll += '        rm tmp.txt\n'
        ll += '        index=`echo "$index+1" | bc -l`\n'
        ll += '        echo "NEW `date` --> JOBSEND: ${index}, TOTALJOBS: ${TOTALJOBS}, ${base}"\n'
        ll += '    else\n'
        ll += '        # Each 60 seconds checks the jobs\n'
        ll += '        sleep 60\n'
        ll += '        echo  "WAIT `date` --> JOBSEND: ${JOBSEND}, TOTALJOBS: ${TOTALJOBS}"\n'
        ll += '    fi\n'
        ll += '\n'
        ll += '    NJOBS=`squeue -h |wc -ll`\n'
        ll += '    TOTALJOBS=`ls -ld ./*_[0-9]*com |wc -ll`\n'
        ll += 'done\n'
        ll += '\necho \'Jobs Done!!!!\'\n'

        txt = ll

        f.writelines(txt)


# ===========================================================================================
def plot_data(df):

    plt.style.use("bmh")
    # ax = df.plot.scatter(x='psi_deg', y='phi_deg')
    # plt.show()
    #
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111,projection='3d')
    #
    # cb = ax.scatter(df.psi_deg, df.phi_deg, df.deltaE_kcalmol, cmap='coolwarm')
    # plt.colorbar(cb)
    # plt.show()

    surf = ax.plot_wireframe(df.psi, df.phi, np.array((df.deltaE_kcalmol,df.deltaE_kcalmol)), rstride=10, cstride=10)
    plt.show()


# ===========================================================================================
def main_app(version):

    #
    args = parse_arguments()
    dir_loc = os.path.join(args.dir, "*.log")
    list_log = sorted(glob.glob(dir_loc))

    if len(list_log) == 0:
        print("ERROR. Log files cannot be found in {}".format(args.dir))
        exit()

    # DataFrame
    df = pd.DataFrame({'ifile': [],
                       'Energy_Ha': [],})
    for item in args.names:
        df[item+"0"] = []
        df[item] = []

    df['deltaE_kcalmol'] = []

    for irow, ifile in enumerate(list_log):
        scf_energy = extract_energy_frozen(ifile, "SCF Done:")
        df.loc[irow] = pd.Series({"Energy_Ha": scf_energy[0], "ifile": os.path.split(ifile)[-1]})
        for jdx, ilabel in enumerate(args.labels):
            angle = extract_energy_frozen(ifile, args.labels[jdx])
            df.loc[irow][args.names[jdx]+"0"] = angle[0]
            df.loc[irow][args.names[jdx]] = angle[-1]

    minscf = df['Energy_Ha'].min()
    maxscf = df['Energy_Ha'].max()

    for irow, row in df.iterrows():
        df.loc[irow]['deltaE_kcalmol'] = (df.loc[irow]['Energy_Ha'] - minscf) * 627.5096

    df.drop(df[df['deltaE_kcalmol'] >= args.dropenergy].index, inplace=True)
    try:
        plot_data(df)
    except AttributeError:
        pass
    print(df.sort_values(by=['deltaE_kcalmol']))
    generate_bashscript_send_slurm()

    dfhtml = df.to_html()
    if args.dir[-1] == "/":
        args.dir = args.dir[0:-1]
    pattern = os.path.split(args.dir)[-1]
    with open(pattern + ".html", 'w') as fhtml:
        fhtml.writelines(dfhtml)


    # df.drop(df[df['deltaE_kcalmol'] >= 9.0].index, inplace=True)
    # # print(df.count(axis=0))
    # # ifile = (df['deltaE_kcalmol']) >= 9.0
    # plot_data(df)
    # print(df.sort_values(by=['deltaE_kcalmol']))
    # #df.to_excel('data12.xlsx')
    # generate_bashscript_send_slurm()


# =============================================================================
if __name__ == "__main__":
    __version__ = "1.1"
    main_app(__version__)
    print("Job done.!!!!")
