#include "arguproc.h"

/* parse command line arguments */
void ArgumentProcess(int argc, char **argv, ARGUTYPE *parg)
{
  int c;

  parg->big_sac = NULL;
  parg->stack_dir = NULL;
  parg->gpu_id = 0;
  parg->save_linear = 1;
  parg->save_pws = 0;
  parg->save_tfpws = 0;
  parg->gpu_task_num = 1;

  /* check argument */
  if (argc <= 1)
  {
    // usage();
    exit(-1);
  }

  /* new stype parsing command line options */
  while ((c = getopt(argc, argv, "I:O:G:S:U:")) != -1)
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
        fprintf(stderr, "Error: Option -S requires a four-digit binary number consisting of 0s and 1s.\n");
        exit(-1);
      }
      parg->save_linear = (optarg[0] == '1') ? 1 : 0; // 解析第一位
      parg->save_pws = (optarg[1] == '1') ? 1 : 0;    // 解析第二位
      parg->save_tfpws = (optarg[2] == '1') ? 1 : 0;  // 解析第三位
      break;
    case 'U': // 新增的解析
      parg->gpu_task_num = atoi(optarg);
      if (parg->gpu_task_num < 1)
      {
        fprintf(stderr, "Error: -N must be >= 1.\n");
        exit(-1);
      }
      break;
    case '?':
    default:
      fprintf(stderr, "Unknown option %c\n", optopt);
      exit(-1);
    }
  }

  /* end of parsing command line arguments */
}

void usage()
{
  fprintf(
      stderr,
      "\nUsage:\n"
      "specxc_mg -I bigsac -O output_dir -G gpu_id -S save_option\n"
      "Options:\n"
      "    -I the Big SAC File computed from Cross-Correlation\n"
      "    -O Specify the output directory for NCF files as sac format\n"
      "    -G ID of Gpu device to be launched \n"
      "    -S Save options: 3 digits binary number, 1 for save, 0 for not save for [linear,pws,tfpws]\n"
      "    -U <int>     Number of tasks to run concurrently on the GPU.\n"
      "Version:\n"
      "  last update by wangjx@20250403\n"
      "  cuda version\n");
}