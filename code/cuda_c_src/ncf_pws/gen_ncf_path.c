#include "gen_ncf_path.h"
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

char *my_strdup(const char *s)
{
    size_t len = strlen(s) + 1;
    char *copy = malloc(len);
    if (copy)
    {
        memcpy(copy, s, len);
    }
    return copy;
}

/* Split the file name */
void SplitFileName(const char *fname, const char *delimiter, char *sta_pair_str, char *cmp_pair_str, char *suffix)
{
    if (!fname || !delimiter || !sta_pair_str || !cmp_pair_str || !suffix)
    {
        return; // check parameters
    }

    char *fname_copy = my_strdup(fname); // in oder not to change the original fname
    char *saveptr;

    char *result = strtok_r(fname_copy, delimiter, &saveptr);
    if (result)
    {
        strcpy(sta_pair_str, result);
    }
    else
    {
        goto cleanup;
    }

    result = strtok_r(NULL, delimiter, &saveptr);
    if (result)
    {
        strcpy(cmp_pair_str, result);
    }
    else
    {
        goto cleanup;
    }

    result = strtok_r(NULL, delimiter, &saveptr);
    if (result)
    {
        strcpy(suffix, result);
    }
    else
    {
        goto cleanup;
    }

cleanup:
    free(fname_copy); // FREE MEMORY
}