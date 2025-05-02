#include "arguproc.h"

/*-------------------------------------------------------------
 * Parse command-line arguments
 *-----------------------------------------------------------*/
void ArgumentProcess(int argc, char **argv, ARGUTYPE *parg)
{
  /* ---------- defaults ---------- */
  parg->big_sac = NULL;
  parg->stack_dir = NULL;
  parg->gpu_id = 0;
  parg->save_linear = 1;
  parg->save_pws = 0;
  parg->save_tfpws = 0;
  parg->gpu_task_num = 1;
  parg->sub_stack_size = 1;   /* 1 (or 0) ⇒ feature disabled */
  parg->src_info_file = NULL; /* 默认不做源信息过滤          */

  if (argc <= 1)
  {
    usage();
    exit(EXIT_FAILURE);
  }

  int c;
  while ((c = getopt(argc, argv, "I:O:G:S:U:B:F:")) != -1)
  {
    switch (c)
    {
    case 'I':
      parg->big_sac = optarg;
      break;
    case 'O':
      parg->stack_dir = optarg;
      break;
    case 'G':
      parg->gpu_id = atoi(optarg);
      break;

    case 'S':
      if (strlen(optarg) != 3 || strspn(optarg, "01") != 3)
      {
        fprintf(stderr,
                "Error: -S expects a 3-digit binary string, e.g., 110.\n");
        exit(EXIT_FAILURE);
      }
      parg->save_linear = (optarg[0] == '1');
      parg->save_pws = (optarg[1] == '1');
      parg->save_tfpws = (optarg[2] == '1');
      break;

    case 'U':
      parg->gpu_task_num = atoi(optarg);
      if (parg->gpu_task_num < 1)
      {
        fprintf(stderr, "Error: -U must be ≥ 1.\n");
        exit(EXIT_FAILURE);
      }
      break;

    case 'B': /* sub-stack size */
      parg->sub_stack_size = atoi(optarg);
      if (parg->sub_stack_size < 2)
        parg->sub_stack_size = 1; /* keep “disabled” state as 1 */
      break;

    case 'F': /* source-info file */
      parg->src_info_file = optarg;
      break;

    default:
      fprintf(stderr, "Unknown option %c\n", optopt);
      usage();
      exit(EXIT_FAILURE);
    }
  }
}

/* 打印用法 */
void usage(void)
{
  fprintf(stderr,
          "\nUsage:\n"
          "  ncf_pws -I <big_sac> -O <out_dir> [options]\n\n"
          "Required arguments:\n"
          "  -I <file>   Big SAC file containing multiple traces\n"
          "  -O <dir>    Output directory for stack results\n\n"
          "Optional arguments:\n"
          "  -G <int>    GPU device ID to use (default: 0)\n"
          "  -S <bin>    Three-digit binary flags to save outputs [linear, pws, tfpws]\n"
          "              e.g. 111 = save all, 100 = save linear only (default: 100)\n"
          "  -U <int>    Number of concurrent GPU tasks (default: 1)\n"
          "  -B <int>    Sub-stack size: pre-stack every <int> traces before PWS/TF-PWS\n"
          "  -F <file>   Source-info file; only traces listed here will be stacked (optional)\n\n"
          "              Set 1 to disable this feature (default: disabled)\n\n"
          "Version: 2025-05-02\n");
}
