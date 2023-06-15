from os.path import join 
from glob import glob
from argparse import ArgumentParser
from soundfile import read
from tqdm import tqdm
from pesq import pesq
import pandas as pd

from pystoi import stoi

from sgmse.util.other import energy_ratios, mean_std


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the original test data (must have subdirectories clean/ and noisy/)')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    args = parser.parse_args()

    test_dir = args.test_dir
    clean_dir = join(test_dir, "s1/")
    noisy_dir = join(test_dir, "mix_clean/")
    enhanced_dir = args.enhanced_dir

    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [],  "si_sar": [], "si_sdr_i": [], "si_sir_i": [],  "si_sar_i": []}
    sr = 16000

    # Evaluate standard metrics   
    enhanced_files = sorted(glob('{}/*_use_vae_predmel0_200steps.wav'.format(enhanced_dir)))
    print("len(enhanced_files): ", len(enhanced_files))
    for enhanced_file in tqdm(enhanced_files):
        filename = enhanced_file.split('/')[-1].split("_use")[0]+".wav"
        x, _ = read(join(clean_dir, filename))
        y, _ = read(join(noisy_dir, filename))

        
        x_method, _ = read(enhanced_file)
        min_len = min(len(x), len(y),len(x_method))
        
        x = x[:min_len]
        y = y[:min_len]
        x_method = x_method[:min_len]
        n = y - x 

        data["filename"].append(filename)
        data["pesq"].append(pesq(sr, x, x_method, 'wb'))
        data["estoi"].append(stoi(x, x_method, sr, extended=True))
        data["si_sdr"].append(energy_ratios(x_method, x, n)[0])
        data["si_sir"].append(energy_ratios(x_method, x, n)[1])
        data["si_sar"].append(energy_ratios(x_method, x, n)[2])
        data["si_sdr_i"].append(energy_ratios(x_method, x, n)[0] - energy_ratios(y, x, n)[0])
        data["si_sir_i"].append(energy_ratios(x_method, x, n)[1] - energy_ratios(y, x, n)[1])
        data["si_sar_i"].append(energy_ratios(x_method, x, n)[2] - energy_ratios(y, x, n)[2])
        

    # Save results as DataFrame    
    df = pd.DataFrame(data)

    # POLQA evaluation  -  requires POLQA license and server, uncomment at your own peril.
    # This is batch processed for speed reasons and thus runs outside the for loop.
    # if not basic:
    #     clean_files = sorted(glob('{}/*.wav'.format(clean_dir)))
    #     enhanced_files = sorted(glob('{}/*.wav'.format(enhanced_dir)))
    #     clean_audios = [read(clean_file)[0] for clean_file in clean_files]
    #     enhanced_audios = [read(enhanced_file)[0] for enhanced_file in enhanced_files]
    #     polqa_vals = polqa(clean_audios, enhanced_audios, 16000, save_to=None)
    #     polqa_vals = [val[1] for val in polqa_vals]
    #     # Add POLQA column to DataFrame
    #     df['polqa'] = polqa_vals

    # Print results
    print(enhanced_dir)
    #print("POLQA: {:.2f} ± {:.2f}".format(*mean_std(df["polqa"].to_numpy())))
    print("PESQ: {:.2f} ± {:.2f}".format(*mean_std(df["pesq"].to_numpy())))
    print("ESTOI: {:.2f} ± {:.2f}".format(*mean_std(df["estoi"].to_numpy())))
    print("SI-SDR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sdr"].to_numpy())))
    print("SI-SIR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sir"].to_numpy())))
    print("SI-SAR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sar"].to_numpy())))
    print("SI-SDR_I: {:.1f} ± {:.1f}".format(*mean_std(df["si_sdr_i"].to_numpy())))
    print("SI-SIR_I: {:.1f} ± {:.1f}".format(*mean_std(df["si_sir_i"].to_numpy())))
    print("SI-SAR_I: {:.1f} ± {:.1f}".format(*mean_std(df["si_sar_i"].to_numpy())))

    # Save DataFrame as csv file
    df.to_csv(join(enhanced_dir, "_results.csv"), index=False)
