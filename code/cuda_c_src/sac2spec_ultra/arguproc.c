#include "arguproc.h"

void ArgumentProcess(int argc, char **argv, ARGUTYPE *pargument)
{
    if (argc < 2)
    {
        usage();
        exit(1);
    }
    int opt;

    // Set default values for optional arguments
    pargument->whitenType = 0;
    pargument->normalizeType = 0;
    pargument->skip_step_count = 0;
    pargument->thread_num = 1;
    pargument->segshift = 0.0f;

    // Parse command line arguments using getopt
    while ((opt = getopt(argc, argv, "I:O:C:B:L:G:F:W:N:Q:T:U:S:")) != -1)
    {
        switch (opt)
        {
        case 'I':
            pargument->sac_lst = optarg;
            break;
        case 'O':
            pargument->spec_lst = optarg;
            break;
        case 'C':
            pargument->num_ch = atoi(optarg);
            break;
        case 'B':
            pargument->filter_file = optarg;
            break;
        case 'L':
            pargument->seglen = atof(optarg);
            break;
        case 'S':
            pargument->segshift = atof(optarg);
            break;

        case 'G':
            pargument->gpu_id = atoi(optarg);
            break;
        case 'F':
        {
            float freq_low, freq_high;
            if (sscanf(optarg, "%f/%f", &freq_low, &freq_high) != 2)
            {
                fprintf(stderr, "Error: Invalid frequency band format\n");
                exit(1);
            }

            // ensure freq_low_limit < freq_high_limit
            if (freq_low >= freq_high)
            {
                fprintf(stderr, "Error: Invalid frequency band range\n");
                exit(1);
            }
            pargument->freq_low = freq_low;
            pargument->freq_high = freq_high;
            break;
        }
        case 'W':
            pargument->whitenType = atoi(optarg);
            break;
        case 'N':
            pargument->normalizeType = atoi(optarg);
            break;
        case 'Q':
        {
            char *token = strtok(optarg, "/");
            pargument->skip_step_count = 0;
            while (token != NULL && pargument->skip_step_count + 1 < MAX_SKIP_STEPS_SIZE)
            {
                int val = atoi(token);
                if (val == -1)
                {
                    break; // stop parsing when find -1
                }
                pargument->skip_steps[pargument->skip_step_count++] = val;
                token = strtok(NULL, "/");
            }
            break;
        }
        case 'U':
            pargument->gpu_num = atoi(optarg);
            break;
        case 'T':
            pargument->thread_num = atoi(optarg);
            break;
        default: // '?' or ':' for unrecognized options or missing option arguments
            usage();
            exit(-1);
        }
    }
}

/* display usage */
void usage(void)
{
    fprintf(stderr,
        "Usage:  sac2spec  -I <sac_list> -O <spec_list> -L <win_len_sec>  [options]\n"
        "\n"
        "Required arguments\n"
        "  -I FILELIST        List of input SAC files (single- or multi-component)\n"
        "  -O FILELIST        List that will receive output segment spectra (1-to-1 with -I)\n"
        "  -L FLOAT           Segment-window length in seconds\n"
        "  -S FLOAT           Segment shift\n"
        "  -C INT             Number of channels in each SAC file         (default: 1)\n"
        "  -G INT             GPU device index to use                     (default: 0)\n"
        "  -U INT             Number of GPUs to launch in parallel        (default: 1)\n"
        "  -T INT             Number of CPU threads                       (default: 1)\n"
        "  -B FILE            Butterworth filter-coefficient file\n"
        "  -F F1/F2           Whitening-band limits in Hz, e.g. 0.01/10\n"
        "  -W INT             Whitening strategy (default: 0)\n"
        "                       0  none\n"
        "                       1  before time-domain normalisation\n"
        "                       2  after  time-domain normalisation\n"
        "                       3  both before and after normalisation\n"
        "  -N INT             Time-domain normalisation type (default: 0)\n"
        "                       0  none\n"
        "                       1  run-abs\n"
        "                       2  one-bit\n"
        "  -Q LIST            Slash-separated list of *segment indices* to skip, e.g. \"3/7/9\";\n"
        "                     terminate with -1. Empty list ⇒ process all segments.\n"
        "  -h                 Show this help and exit\n"
        "\n"
        "Example\n"
        "  sac2spec -I sac.lst -O spec.lst -L 30 -S 15 -C 3 -F 0.01/10 \\\n"
        "          -W 1 -N 1 -B bw_coeff.txt -G 0 -U 1 -T 4 -Q 5/10/-1\n"
        "\n"
        "Last updated: 2025-04-21  (wangjx)\n");
}
