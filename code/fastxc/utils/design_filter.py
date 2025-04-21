import os
from scipy.signal import butter
import logging

logger = logging.getLogger(__name__)


def design_filter(delta, bands, output_file):
    """
    Design filter for fastxc, save the filter coefficients to file.

    Parameters
    ----------
    delta : float
    bands : str
    output_file : str

    Returns
    -------
    None
    """
    # check ouput directory, if not exist, create it
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Output directory '{output_dir}' has been created.")

    fs = 1.0 / delta  # sampling frequency
    f_nyq = fs / 2.0  # Nyquist frequency
    order = 2  # filter order

    bands_str = bands.split()
    all_freqs = []
    for band_str in bands_str:
        freq_low, freq_high = map(float, band_str.split("/"))
        all_freqs.append((freq_low, freq_high))

    # Get the overall min and max frequencies
    overall_min = min(freq[0] for freq in all_freqs)
    overall_max = max(freq[1] for freq in all_freqs)

    # Check if the overall band is valid
    if not (0 < overall_min < overall_max < f_nyq):
        logger.error(f"Invalid overall band: {overall_min}/{overall_max}")
        raise ValueError

    # Normalize frequencies
    overall_min_norm = overall_min / f_nyq
    overall_max_norm = overall_max / f_nyq

    # Design the overall bandpass filter
    b, a = butter(order, [overall_min_norm, overall_max_norm], btype="bandpass")

    # Write filters to file
    try:
        with open(output_file, "w") as f:
            # Write the overall filter first
            f.write(f"# {overall_min}/{overall_max}\n")
            f.write("\t".join(f"{b_i:.18e}" for b_i in b) + "\n")
            f.write("\t".join(f"{a_i:.18e}" for a_i in a) + "\n")

            # Now write the individual band filters
            for band_str in bands_str:
                freq_low, freq_high = map(float, band_str.split("/"))
                freq_low_norm = freq_low / f_nyq
                freq_high_norm = freq_high / f_nyq
                b, a = butter(order, [freq_low_norm, freq_high_norm], btype="bandpass")

                line_b = "\t".join(f"{b_i:.18e}" for b_i in b)
                line_a = "\t".join(f"{a_i:.18e}" for a_i in a)

                f.write(f"# {band_str}\n")
                f.write(line_b + "\n")
                f.write(line_a + "\n")
    except IOError as e:
        logger.error(f"Error writing filter to file: {e}")


if __name__ == "__main__":
    design_filter(
        {
            "bands": "0.2/0.5 0.6/0.8",  # define frequency bands
            "output_file": "./filter.txt",  # define output directory
            "delta": 0.01,  # define sampling interval
        }
    )
