#include "arguproc.h"

/* parse command line arguments */
void ArgumentProcess(int argc, char **argv, ARGUTYPE *parg)
{
  int c;

  /* 初始化默认值 */
  parg->src_lst_path = NULL;
  parg->sta_lst_path = NULL;
  parg->ncf_dir = NULL;
  parg->cclength = 0.0f;

  parg->distmin = 0.0f;      /* 默认下限距离 */
  parg->distmax = 400000.0f; /* 默认上限距离 */

  parg->azmin = 0.0f;   /* 默认下限方位角 */
  parg->azmax = 360.0f; /* 默认上限方位角 */

  parg->srcinfo_file = NULL;

  parg->gpu_id = 0;
  parg->queue_id = 0; /* 新增字段，默认0 */
  parg->gpu_task_num = 1;
  parg->cpu_count = 1;
  parg->write_mode = 1; /* 1=APPEND */

  if (argc <= 1)
  {
    usage();
    exit(-1);
  }

  /* 解析命令行选项 */
  while ((c = getopt(argc, argv, "A:B:O:C:G:U:T:D:M:S:Z:Q:")) != -1)
  {
    switch (c)
    {
    case 'A':
      parg->src_lst_path = optarg;
      break;

    case 'B':
      parg->sta_lst_path = optarg;
      break;

    case 'O':
      parg->ncf_dir = optarg;
      break;

    case 'C':
      parg->cclength = (float)atof(optarg);
      break;

    /*
     * 旧版 -G X/Y 改为新版只解析 GPU ID，例如:
     *   -G 0  ==> gpu_id=0
     */
    case 'G':
      parg->gpu_id = (size_t)atoi(optarg);
      break;

    /* 新增 -Q <queue_id> */
    case 'Q':
      parg->queue_id = (size_t)atoi(optarg);
      break;

    case 'U':
      parg->gpu_task_num = (size_t)atoi(optarg);
      break;

    case 'T':
      parg->cpu_count = (size_t)atoi(optarg);
      break;

    case 'M':
      parg->write_mode = atoi(optarg);
      if (parg->write_mode < 0 || parg->write_mode > 2)
      {
        fprintf(stderr,
                "Invalid write mode %d, should be:\n"
                "  0 -> traditional\n"
                "  1 -> append\n"
                "  2 -> aggregate\n",
                parg->write_mode);
        exit(-1);
      }
      break;

    case 'S':
      parg->srcinfo_file = optarg;
      break;

    case 'Z':
    {
      /* 解析方位角范围，如 10/360 */
      char *slash = strchr(optarg, '/');
      if (slash == NULL)
      {
        fprintf(stderr, "Invalid format for -Z, should be AZMIN/AZMAX.\n");
        exit(-1);
      }
      *slash = '\0';
      parg->azmin = (float)atof(optarg);
      parg->azmax = (float)atof(slash + 1);
      break;
    }

    case 'D':
    {
      /*
       * 解析距离范围，如 50/400
       */
      char *slash = strchr(optarg, '/');
      if (slash == NULL)
      {
        fprintf(stderr, "Invalid format for -D, should be DISTMIN/DISTMAX.\n");
        exit(-1);
      }
      *slash = '\0';
      parg->distmin = (float)atof(optarg);
      parg->distmax = (float)atof(slash + 1);
      break;
    }

    case '?':
    default:
      fprintf(stderr, "Unknown option: -%c\n", optopt);
      exit(-1);
    }
  } /* end while */
}

void usage()
{
  fprintf(stderr,
          "\nUsage:\n"
          "  specxc_mg [OPTIONS]\n"
          "\n"
          "Options:\n"
          "  -A <file>            List file of input for the 1st station (virtual source)\n"
          "  -B <file>            List file of input for the 2nd station (virtual station)\n"
          "  -O <dir>             Output directory for NCF in SAC format\n"
          "  -C <seconds>         Half of CC length (in seconds)\n"
          "  -S <file>            Source info file\n"
          "  -Z <azmin/azmax>     Azimuth range in degrees\n"
          "  -D <distmin/distmax> Distance range in km\n"
          "\n"
          "  -G <gpu_id>          GPU device ID (e.g., 0)\n"
          "  -Q <queue_id>        Queue ID\n"
          "  -U <num>             Number of tasks deployed on a single GPU\n"
          "  -T <num>             Number of CPU threads to use\n"
          "  -M <mode>            Write mode:\n"
          "                       0 -> traditional\n"
          "                       1 -> append\n"
          "                       2 -> aggregate\n"
          "\n"
          "Example:\n"
          "  specxc_mg -A src.lst -B sta.lst -O out -C 50 \\\n"
          "            -S source_info.txt -Z 10/360 -D 0/400 \\\n"
          "            -G 0 -Q 1 -U 4 -T 8 -M 1\n"
          "\n"
          "Version:\n"
          "  last update by ChatGPT & wjx @20250402\n"
          "  cuda version\n");
}
