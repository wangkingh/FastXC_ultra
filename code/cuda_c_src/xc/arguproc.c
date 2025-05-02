#include "arguproc.h"

void ArgumentProcess(int argc, char **argv, ARGUTYPE *parg)
{
  int c;

  /* ---------- 默认值 ---------- */
  parg->src_lst_path = parg->sta_lst_path = parg->ncf_dir = NULL;
  parg->cclength = 0.0f;

  parg->distmin = 0.0f;
  parg->distmax = 400000.0f;

  parg->azmin = 0.0f;
  parg->azmax = 360.0f;

  parg->srcinfo_file = NULL;

  parg->gpu_id = 0;
  parg->queue_id = 0;
  parg->gpu_task_num = 1;
  parg->cpu_count = 1;
  parg->write_mode = 1; /* 1: append; 2:aggregate */
  parg->save_segment = 0;   /* 默认走叠加+IFFT,不保存每一段的结果 */

  if (argc <= 1)
  {
    usage();
    exit(-1);
  }

  /* ---------- 解析选项 ---------- */
  /* 注意：R 无冒号  -> 无参数 */
  while ((c = getopt(argc, argv,
                     "A:B:O:C:G:U:T:D:M:S:Z:Q:R:")) != -1)
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
      parg->cclength = atof(optarg);
      break;

    case 'G':
      parg->gpu_id = (size_t)atoi(optarg);
      break;
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
                "Invalid write mode %d (0|1|2).\n",
                parg->write_mode);
        exit(-1);
      }
      break;

    case 'S': /* 源信息文件 */
      parg->srcinfo_file = optarg;
      break;

    case 'R': /* 保存频域开关 */
      parg->save_segment = atoi(optarg);
      break;

    /* ---------- 复合参数 ---------- */
    case 'Z':
    { /* azmin/azmax */
      char *slash = strchr(optarg, '/');
      if (!slash)
      {
        fprintf(stderr, "Bad -Z format.\n");
        exit(-1);
      }
      *slash = '\0';
      parg->azmin = atof(optarg);
      parg->azmax = atof(slash + 1);
      break;
    }

    case 'D':
    { /* distmin/distmax */
      char *slash = strchr(optarg, '/');
      if (!slash)
      {
        fprintf(stderr, "Bad -D format.\n");
        exit(-1);
      }
      *slash = '\0';
      parg->distmin = atof(optarg);
      parg->distmax = atof(slash + 1);
      break;
    }

    default:
      fprintf(stderr, "Unknown option -%c\n", optopt);
      exit(-1);
    }
  }
}

void usage(void)
{
  fprintf(stderr,
          "\nUsage:\n"
          "  specxc_mg [OPTIONS]\n\n"
          "Core I/O:\n"
          "  -A <file>        List of source spectra\n"
          "  -B <file>        List of station spectra\n"
          "  -O <dir>         Output directory (NCF/SAC)\n"
          "  -C <sec>         Half CC length\n"
          "  -S <file>        Source-info file (optional)\n"
          "  -R               Output raw segments (skip step-sum)\n\n"
          "Selection:\n"
          "  -Z <azmin/azmax> Azimuth range (deg)\n"
          "  -D <dmin/dmax>   Distance range (km)\n\n"
          "Runtime:\n"
          "  -G <gpu>         GPU ID\n"
          "  -Q <queue>       Queue ID\n"
          "  -U <num>         Tasks per GPU\n"
          "  -T <num>         CPU threads\n"
          "  -M <0|1|2>       Write mode (trad/append/aggregate)\n\n"
          "Example:\n"
          "  specxc_mg -A src.lst -B sta.lst -O out -C 50 \\\n"
          "            -S srcinfo.txt -Z 10/360 -D 0/400 -R \\\n"
          "            -G 0 -Q 1 -U 4 -T 8 -M 1\n\n");
}
