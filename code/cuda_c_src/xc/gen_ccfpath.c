#include "gen_ccfpath.h"

// Recursively creates directories
void CreateDir(char *sPathName)
{
    char DirName[512];
    strcpy(DirName, sPathName);
    int i, len = strlen(DirName);
    for (i = 1; i < len; i++)
    {
        if (DirName[i] == '/')
        {
            DirName[i] = '\0';
            if (access(DirName, 0) != 0)
            {
                if (mkdir(DirName, 0755) == -1)
                {
                    // Check again if the directory exists after the error
                    if (access(DirName, 0) != 0)
                    {
                        printf("[INFO] Error creating %s. Permission denied\n", DirName);
                    }
                }
            }
            DirName[i] = '/';
        }
    }
    if (len > 0 && access(DirName, 0) != 0)
    {
        if (mkdir(DirName, 0755) == -1)
        {
            // Check again if the directory exists after the error
            if (access(DirName, 0) != 0)
            {
                printf("[INFO] Error creating %s. Permission denied\n", DirName);
            }
        }
    }
}

/* Split the file name */
void SplitFileName(const char *fname, const char *delimiter, char *stastr,
                   char *yearstr, char *jdaystr, char *hmstr, char *chnstr)
{
    if (!fname || !delimiter || !stastr || !yearstr || !jdaystr || !hmstr || !chnstr)
    {
        return; // check parameters
    }

    char *fname_copy = my_strdup(fname); // in oder not to change the original fname
    char *saveptr;

    char *result = strtok_r(fname_copy, delimiter, &saveptr);
    if (result)
    {
        strcpy(stastr, result);
    }
    else
    {
        goto cleanup;
    }

    result = strtok_r(NULL, delimiter, &saveptr);
    if (result)
    {
        strcpy(yearstr, result);
    }
    else
    {
        goto cleanup;
    }

    result = strtok_r(NULL, delimiter, &saveptr);
    if (result)
    {
        strcpy(jdaystr, result);
    }
    else
    {
        goto cleanup;
    }

    result = strtok_r(NULL, delimiter, &saveptr);
    if (result)
    {
        strcpy(hmstr, result);
    }
    else
    {
        goto cleanup;
    }

    result = strtok_r(NULL, delimiter, &saveptr);
    if (result)
    {
        strcpy(chnstr, result);
    }

cleanup:
    free(fname_copy); // FREE MEMORY
}

void SacheadProcess(SACHEAD *ncfhd,
                    float stla, float stlo, float evla, float evlo,
                    float Gcarc, float Az, float Baz, float Dist,
                    float delta, int ncc, float cclength, TimeData *time_info)
{
    *ncfhd = sac_null;
    /* Write in stla,stlo,evla,evlo*/
    ncfhd->stla = stla;
    ncfhd->stlo = stlo;
    ncfhd->evla = evla;
    ncfhd->evlo = evlo;

    // Convert back to float after the function call
    ncfhd->gcarc = Gcarc;
    ncfhd->az = Az;
    ncfhd->baz = Baz;
    ncfhd->dist = Dist;

    /* necessary header info */
    ncfhd->iftype = 1;
    ncfhd->leven = 1;
    ncfhd->delta = delta;
    /* npts of hd should b ncfnpts+1, eg 2k+1 */
    ncfhd->npts = ncc;
    ncfhd->b = -1.0 * cclength;
    ncfhd->e = cclength;
    ncfhd->unused27 = 1;
    /* set o time to be zero */
    ncfhd->o = 0.0;

    // 5) Set the time info
    if (time_info)
    {
        ncfhd->nzyear = time_info->year;
        ncfhd->nzjday = time_info->day_of_year;
        ncfhd->nzhour = time_info->hour;
        ncfhd->nzmin = time_info->minute;
        ncfhd->nzsec = 0;
        ncfhd->nzmsec = 0;
    }

    /* END OF MAKE COMMON HEADER INFO */
}

/* Generate the output XC(Cross-Correlation) path*/
void GenCCFPath(char *ccf_path, char *src_path, char *sta_path, char *output_dir, size_t queue_id)
{
    // Extract file names from source and station paths
    char src_file_name[MAXNAME];
    char sta_file_name[MAXNAME];

    char src_station[16], src_channel[16];
    char sta_station[16], sta_channel[16];

    char src_year[5], src_jday[4], src_hm[5];
    char sta_year[5], sta_jday[4], sta_hm[5];

    strncpy(src_file_name, basename(src_path), MAXNAME);
    strncpy(sta_file_name, basename(sta_path), MAXNAME);

    // Split file names into individual components
    SplitFileName(src_file_name, ".", src_station, src_year, src_jday, src_hm, src_channel);
    SplitFileName(sta_file_name, ".", sta_station, sta_year, sta_jday, sta_hm, sta_channel);
    char ccf_dir[MAXLINE];
    char ccf_name[MAXNAME];

    // Generate output directory and CCF path
    snprintf(ccf_dir, sizeof(ccf_dir), "%s/queue_%li", output_dir, queue_id);

    // Generate new CCF file name based on time cross flag
    snprintf(ccf_name, MAXLINE, "%s-%s.%s-%s.bigsac", src_station, sta_station, src_channel, sta_channel);

    CreateDir(ccf_dir);
    snprintf(ccf_path, 2 * MAXLINE, "%s/%s", ccf_dir, ccf_name);
}
